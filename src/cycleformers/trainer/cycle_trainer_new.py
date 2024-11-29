"""Completely fresh attempt, based on the huggingface trainer but not using it for now"""
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorWithPadding, Trainer, DataCollatorForLanguageModeling
from transformers.training_args import TrainingArguments
from functools import wraps
import inspect
from datasets import load_from_disk, Dataset
from dataclasses import dataclass
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys

from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.training_args import TrainingArguments
from functools import wraps
import inspect


def auto_temp_attributes(*attrs_to_cleanup):
    """
    Decorator that automatically sets and manages temporary attributes on a class instance.
    This solves the issue of methods that modify attributes that are needed for other methods.
    Parameters matching attribute names are set to their passed values, others to None.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Store original attribute values
            original_values = {}
            for attr in attrs_to_cleanup:
                if hasattr(self, attr):
                    original_values[attr] = getattr(self, attr)
            
            # Get function signature to match positional args to parameter names
            sig = inspect.signature(func)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()
            
            # Set attributes based on parameters
            for attr in attrs_to_cleanup:
                # Skip 'self' parameter
                if attr in bound_args.arguments and attr != 'self':
                    setattr(self, attr, bound_args.arguments[attr])
                else:
                    setattr(self, attr, None)
            
            try:
                # Execute the method
                result = func(self, *args, **kwargs)
                return result
            finally:
                # Clean up attributes
                for attr in attrs_to_cleanup:
                    if attr in original_values:
                        setattr(self, attr, original_values[attr])
                    else:
                        try:
                            delattr(self, attr)
                        except AttributeError:
                            pass
        return wrapper
    return decorator


@dataclass
class CycleTrainingArguments(TrainingArguments):
    pass

@dataclass
class CycleStepParams:
    cycle_name: str
    idx: int
    batch: dict
    model_train: torch.nn.Module
    model_gen: torch.nn.Module
    tokenizer_train: AutoTokenizer
    tokenizer_gen: AutoTokenizer
    optimizer: torch.optim.Optimizer


class CycleTrainer(Trainer):
    def __init__(
            self,
            args,
            model_a = None,
            model_b = None,
            tokenizer_a = None,
            tokenizer_b = None,
            train_dataset_a = None,
            train_dataset_b = None,
            eval_dataset_a = None,
            eval_dataset_b = None,
            data_collator_a = None,
            data_collator_b = None,
            callbacks = None
        ):       
        self.args = args
        self.tokenizer_a = tokenizer_a
        self.tokenizer_b = tokenizer_b
        self.model_a = model_a
        self.model_b = model_b

        if data_collator_a is None:
            data_collator_a = DataCollatorWithPadding(self.tokenizer_a)
        if data_collator_b is None:
            data_collator_b = DataCollatorWithPadding(self.tokenizer_b)

        self.accelerator = Accelerator(log_with="wandb")
        self.accelerator.init_trackers(
            project_name="cycleformers",
            config={},
            init_kwargs={"wandb": {"name": "cycleformers"}}
        
        )
        self.train_dataset_a = train_dataset_a
        self.train_dataset_b = train_dataset_b
        self.eval_dataset_a = eval_dataset_a
        self.eval_dataset_b = eval_dataset_b
        self.data_collator_a = data_collator_a
        self.data_collator_b = data_collator_b

        print("Creating Optimizers")
        optimizer_a, lr_scheduler_a = self.create_optimizer_and_scheduler(model_a, len(train_dataset_a))
        optimizer_b, lr_scheduler_b = self.create_optimizer_and_scheduler(model_b, len(train_dataset_b))

        print("Preparing")
        prepared = self.accelerator.prepare(
            model_a, optimizer_a, lr_scheduler_a, 
            model_b, optimizer_b, lr_scheduler_b
        )
        self.model_a, self.optimizer_a, self.lr_scheduler_a, self.model_b, \
            self.optimizer_b, self.lr_scheduler_b = prepared

    def get_train_dataloader(self, dataset, data_collator):
        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "collate_fn": data_collator
        }
        return self.accelerator.prepare(DataLoader(dataset, **dataloader_params))

    def get_eval_dataloader(self, dataset, data_collator):
        dataloader_params = {
            "batch_size": self.args.per_device_eval_batch_size,
            "collate_fn": data_collator
        }
        return self.accelerator.prepare(DataLoader(dataset, **dataloader_params))

    @auto_temp_attributes('model', 'optimizer', 'lr_scheduler')
    def create_optimizer_and_scheduler(self, model, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps=num_training_steps)
        return self.optimizer, self.lr_scheduler

    def train(self):
        self.train_dataloader_a = self.get_train_dataloader(self.train_dataset_a, self.data_collator_a)
        self.train_dataloader_b = self.get_train_dataloader(self.train_dataset_b, self.data_collator_b)
        total = min(len(self.train_dataloader_a), len(self.train_dataloader_b))
        pbar = tqdm(total=total)

        for idx, (batch_a, batch_b) in enumerate(zip(self.train_dataloader_a, self.train_dataloader_b)):
            self._cycle_step(self.model_a, self.model_b, self.tokenizer_a, self.tokenizer_b, self.optimizer_a, batch_b, idx, cycle_name="A")
            self._cycle_step(self.model_b, self.model_a, self.tokenizer_b, self.tokenizer_a, self.optimizer_b, batch_a, idx, cycle_name="B")
            pbar.update(1)

            if idx % 50 == 0:
                metrics = self.evaluate()
                self.accelerator.log(metrics)

    def _cycle_step(self, model_train, model_gen, tokenizer_train, tokenizer_gen, optimizer, batch, idx, cycle_name):
        model_gen.eval()
        model_train.train()

        step_metrics = {}

        with torch.no_grad():   
            synth_batch = model_gen.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_new_tokens=100)

        # Clean up output for model b - Can be made much more efficient
        synth_batch = tokenizer_gen.batch_decode(synth_batch, skip_special_tokens=True)
        synth_batch = {k: v.to(self.accelerator.device) for k, v in tokenizer_train(synth_batch, return_tensors="pt", padding=True).items()}        

        # Accelerator fails when handling alternating gradients
        outputs = model_train(**synth_batch, labels=batch["input_ids"])
        loss = outputs.loss / self.args.gradient_accumulation_steps
        step_metrics[f"train_loss_{cycle_name}"] = loss
        self.accelerator.backward(loss)

        if (idx+1) % self.args.gradient_accumulation_steps == 0:
            step_metrics[f"lr_{cycle_name}"] = optimizer.param_groups[0]["lr"]
            optimizer.step()
            optimizer.zero_grad()

        self.accelerator.log(step_metrics)
    

    def evaluate(self, ignore_keys=None):
        metrics = {}

        # Evaluate model A
        self.model_a.eval()
        losses_a = []

        eval_dataloader_a = self.get_eval_dataloader(self.eval_dataset_a, self.data_collator_a)
        total_a = len(eval_dataloader_a)
        
        for batch in tqdm(eval_dataloader_a, total=total_a, desc="Evaluating Model A"):
            with torch.no_grad():
                outputs = self.model_a(**batch)
                loss = outputs.loss
                losses_a.append(loss.detach().cpu())
                
        metrics["eval_loss_A"] = torch.mean(torch.tensor(losses_a))

        # Evaluate model B  
        self.model_b.eval()
        losses_b = []

        eval_dataloader_b = self.get_eval_dataloader(self.eval_dataset_b, self.data_collator_b)
        total_b = len(eval_dataloader_b)

        artifacts = []
        for batch in tqdm(eval_dataloader_b, total=total_b, desc="Evaluating Model B"):
            with torch.no_grad():
                outputs = self.model_b(**batch)
                loss = outputs.loss
                losses_b.append(loss.detach().cpu())

        metrics["eval_loss_B"] = torch.mean(torch.tensor(losses_b))

        return metrics

    
def generate_model_samples(model, tokenizer, dataset, data_collator, num_samples=10):
    model.eval()
    samples = []
    for batch in tqdm(dataset, total=num_samples, desc="Generating samples"):
        with torch.no_grad():
            outputs = model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_new_tokens=100)
            samples.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    for sample in samples:
        print(sample)
        print()


            

# ==============================
# This will all be removed after testing

if __name__ == "__main__":
    from transformers import DataCollatorForSeq2Seq
    dataset_en, dataset_de = load_from_disk("data/en"), load_from_disk("data/de")

    model_a = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    model_b = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", padding=True)

    args = CycleTrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8)

    trainer = CycleTrainer(
        args=args,
        model_a=model_a,
        model_b=model_b,
        tokenizer_a=tokenizer,
        tokenizer_b=tokenizer,
        train_dataset_a=dataset_en['train'],
        train_dataset_b=dataset_de['train'],
        eval_dataset_a=dataset_en['test'],
        eval_dataset_b=dataset_de['test'],
        data_collator_a=data_collator,
        data_collator_b=data_collator,
    )
    trainer.train()

    print("English to German")
    generate_model_samples(trainer.model_b, trainer.tokenizer_b, trainer.eval_dataset_a, trainer.data_collator_b)

    print("German to English")
    generate_model_samples(trainer.model_a, trainer.tokenizer_a, trainer.eval_dataset_b, trainer.data_collator_a)