"""Completely fresh attempt, based on the huggingface trainer but not using it for now"""
from contextlib import contextmanager
from typing import Literal
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer
from datasets import load_from_disk, Dataset
from dataclasses import dataclass
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader

import sys
if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback
from transformers.training_args import TrainingArguments

BATCH_SIZE = 4


def check_gradients(model):
    """Helper function to check if gradients exist and their magnitudes"""
    has_grad = False
    grad_mag = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_mag += param.grad.norm().item()
    return has_grad, grad_mag


def check_and_print_grads(model_a, model_b):
    has_grad_a, grad_mag_a = check_gradients(model_a)
    has_grad_b, grad_mag_b = check_gradients(model_b)
    print(f"Model Train has gradients: {has_grad_a}, magnitude: {grad_mag_a:.6f}")
    print(f"Model Gen has gradients: {has_grad_b}, magnitude: {grad_mag_b:.6f}")


class OutOfModelContextError(Exception):
    pass


def protected_attribute(name: str):
    """Decorator factory that creates a property with context manager protection."""
    private_name = f'_{name}'

    error_msg = f"{name} access is not allowed outside of model_context. " \
        "Use 'with trainer.model_context(model_key)' to access protected attributes."
    
    def getter(self):
        if not self._in_context:
            raise OutOfModelContextError(error_msg)
        return getattr(self, private_name)
    
    def setter(self, value):
        if not self._in_context and value is not None:
            raise OutOfModelContextError(error_msg)
        setattr(self, private_name, value)
    
    return property(getter, setter)


@dataclass
class CycleTrainingArguments(TrainingArguments):
    pass


@dataclass
class ModelObjects:
    model: torch.nn.Module
    tokenizer: AutoTokenizer
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler

class CycleTrainer(Trainer):

    MANAGED_ATTRS = ['model', 'optimizer', 'lr_scheduler', 'train_dataset', 'data_collator']

    def __init__(
            self,
            args,
            model_a,
            model_b,
            tokenizer,
            train_dataset_a,
            train_dataset_b,
            eval_dataset_a,
            eval_dataset_b,
            data_collator_a,
            data_collator_b,
            callbacks = None
        ):
        # Initialize managed attributes to use methods that set class attributes directly
        self._in_context = False
        self._original_values = {}
        for attr in self.MANAGED_ATTRS:
            setattr(self, f"_{attr}", None)
            setattr(self.__class__, attr, protected_attribute(attr))
        
        self.args = args
        # TODO: Manually implement gradient accumulation. *2 works only for datasets with even number of batches of batches
        self.accelerator = Accelerator(gradient_accumulation_steps=self.args.gradient_accumulation_steps*2)
        self.tokenizer = tokenizer
        self.data_collator_a = data_collator_a
        self.data_collator_b = data_collator_b
        self.eval_dataset_a = eval_dataset_a
        self.eval_dataset_b = eval_dataset_b
        self.model_a = model_a
        self.model_b = model_b
        self.train_dataset_a = train_dataset_a
        self.train_dataset_b = train_dataset_b
        self.optimizer_a = None
        self.optimizer_b = None
        self.lr_scheduler_a = None
        self.lr_scheduler_b = None
        
        with self.model_context('a'):
            self.create_optimizer_and_scheduler(num_training_steps=len(train_dataset_a)//BATCH_SIZE)
            self.train_dataloader_a = self.get_train_dataloader()

        with self.model_context('b'):
            self.create_optimizer_and_scheduler(num_training_steps=len(train_dataset_b)//BATCH_SIZE)
            self.train_dataloader_b = self.get_train_dataloader()

        # Add everything to accelerator
        self.model_a, self.model_b,
        self.optimizer_a, self.optimizer_b,
        self.lr_scheduler_a, self.lr_scheduler_b, = \
            self.accelerator.prepare(
                self.model_a, self.model_b,
                self.optimizer_a, self.optimizer_b,
                self.lr_scheduler_a, self.lr_scheduler_b,
            )
        
        self.model_a = self.model_a.to(self.accelerator.device)
        self.model_b = self.model_b.to(self.accelerator.device)
        
    @contextmanager
    def model_context(self, model_key: Literal['a', 'b']):
        """Context manager to temporarily swap protected attributes."""
        if self._in_context:
            raise RuntimeError("Nested model contexts are not allowed")
        if model_key not in ['a', 'b']:
            raise ValueError(f"Invalid model_key: {model_key}. Must be 'a' or 'b'")
            
        try:
            # While in the manager, set self.model, etc. to self.model_a, etc.
            self._in_context = True
            for attr in self.MANAGED_ATTRS:
                setattr(self, f'_{attr}', getattr(self, f"{attr}_{model_key}"))
            
            yield
            
        finally:
            # Potential issue could be unwanted setting of value
            for attr in self.MANAGED_ATTRS:
                setattr(self, f"{attr}_{model_key}", getattr(self, f'_{attr}'))
                setattr(self, f'_{attr}', None)
            self._in_context = False


    # def get_train_dataloader(self, dataset):
    #     dataloader_params = {
    #         "batch_size": BATCH_SIZE,
    #         "collate_fn": self.data_collator
    #     }
    #     return self.accelerator.prepare(DataLoader(dataset, **dataloader_params))
    

    # def get_eval_dataloader(self, dataset):
    #     dataloader_params = {
    #         "batch_size": BATCH_SIZE,
    #         "collate_fn": self.data_collator
    #     }
    #     return self.accelerator.prepare(DataLoader(dataset, **dataloader_params))


    def train(self):
        print(self._model)

        for i, (batch_a, batch_b) in enumerate(zip(self.train_dataloader_a, self.train_dataloader_b)):
            print("Cycle B")
            self._cycle_step(self.model_a, self.model_b, self.optimizer_a, batch_b)
            print("Cycle A")
            self._cycle_step(self.model_b, self.model_a, self.optimizer_b, batch_a)
            

    def _cycle_step(self, model_train, model_gen, optimizer, batch):
        model_gen.eval()
        model_train.train()

        with torch.no_grad():   
            synth_batch = model_gen.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_new_tokens=100)

        # Clean up output for model b - Can be made much more efficient
        synth_batch = self.tokenizer.batch_decode(synth_batch, skip_special_tokens=True)
        synth_batch = {k: v.to(self.accelerator.device) for k, v in self.tokenizer(synth_batch, return_tensors="pt", padding=True).items()}        

        # Accelerator fails when handling alternating gradients
        outputs = model_train(**synth_batch, labels=batch["input_ids"])
        # loss = outputs.loss / self.args.gradient_accumulation_steps
        # self.accelerator.backward(loss)

        # if (idx+1) % self.args.gradient_accumulation_steps == 0:
        with self.accelerator.accumulate(model_train, model_gen):
            loss = outputs.loss
            self.accelerator.backward(loss)
            optimizer.step()
            check_and_print_grads(model_train, model_gen)
            optimizer.zero_grad()
            check_and_print_grads(model_train, model_gen)
            print('='*20)
    

    def evaluate(self, ignore_keys=None):
        # First evaluate model a
        eval_dataloader_a = self.get_eval_dataloader(self.eval_dataset_a)
        eval_dataloader_b = self.get_eval_dataloader(self.eval_dataset_b)

        self.model_a.eval()
        
        for batch_a in eval_dataloader_a:
            losses, logits, labels = self.prediction_step(self.model_a, batch_a, ignore_keys=ignore_keys)





            

# ==============================
# This will all be removed after testing

if __name__ == "__main__":
    dataset_en, dataset_de = load_from_disk("data/en"), load_from_disk("data/de")

    model_a = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    model_b = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", padding=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")

    args = CycleTrainingArguments(
        output_dir="outputs",
    )

    trainer = CycleTrainer(
        args,
        model_a,
        model_b,
        tokenizer,
        dataset_en['train'],
        dataset_de['train'],
        dataset_en['test'],
        dataset_de['test'],
        data_collator,
        data_collator
    )
    trainer.train()
