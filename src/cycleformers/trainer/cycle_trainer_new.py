"""Completely fresh attempt, based on the huggingface trainer but not using it for now"""
import math
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorWithPadding, Trainer, AutoModelForCausalLM
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerState, TrainerControl, PrinterCallback, CallbackHandler, ExportableState
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer_utils import TrainOutput
from functools import wraps
import inspect
from datasets import load_from_disk, Dataset
from dataclasses import dataclass
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.training_args import TrainingArguments
from functools import wraps
import inspect

# Need to implement CycleTrainerCallback class that inherits from TrainerCallback
# Should handle both models A and B states

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
            args: CycleTrainingArguments,
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

        # TODO: Check whether seq2seq tokenizers care about padding side
        if tokenizer_a.pad_token is None:
            tokenizer_a.pad_token = tokenizer_a.eos_token
            model_a.generation_config.pad_token_id = tokenizer_a.eos_token_id
            tokenizer_a.padding_side = "left" if not model_a.config.is_encoder_decoder else tokenizer_a.padding_side
        if tokenizer_b.pad_token is None:
            tokenizer_b.pad_token = tokenizer_b.eos_token
            model_b.generation_config.pad_token_id = tokenizer_b.eos_token_id
            tokenizer_b.padding_side = "left" if not model_b.config.is_encoder_decoder else tokenizer_b.padding_side

        self.tokenizer_a = tokenizer_a
        self.tokenizer_b = tokenizer_b

        self.model_a = model_a
        self.model_b = model_b

        if data_collator_a is None and model_a.config.is_encoder_decoder:
            self.data_collator_a = DataCollatorForSeq2Seq(tokenizer_a)
        elif data_collator_a is None:
            self.data_collator_a = DataCollatorWithPadding(tokenizer_a)

        if data_collator_b is None and model_b.config.is_encoder_decoder:
            self.data_collator_b = DataCollatorForSeq2Seq(tokenizer_b)
        elif data_collator_b is None:
            self.data_collator_b = DataCollatorWithPadding(tokenizer_b)

        self.train_dataset_a = train_dataset_a
        self.train_dataset_b = train_dataset_b
        self.eval_dataset_a = eval_dataset_a
        self.eval_dataset_b = eval_dataset_b

        ## Calculate batches 
        if not args.max_steps > 0:
            num_update_steps_per_epoch = min(len(train_dataset_a), len(train_dataset_b)) // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            args.max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
        
        accelerator = Accelerator()
        self.accelerator = accelerator
        args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        args.batch_size = args.local_batch_size
        args.total_num_batches = math.ceil(args.max_steps / args.batch_size)

        ## Setup model and dataloaders
        optimizer_a, lr_scheduler_a = self.create_optimizer_and_scheduler(model_a, args.total_num_batches)
        optimizer_b, lr_scheduler_b = self.create_optimizer_and_scheduler(model_b, args.total_num_batches)

        prepared = self.accelerator.prepare(
            model_a, optimizer_a, lr_scheduler_a, 
            model_b, optimizer_b, lr_scheduler_b
        )
        self.model_a, self.optimizer_a, self.lr_scheduler_a, self.model_b, \
            self.optimizer_b, self.lr_scheduler_b = prepared
        
        ## Trainer specific setup

        # TODO: Major revision of control flow needed for any non-trivial runs
        # Initialise callbacks and control flow - Current strategy is to just track one model
        # under the assumption that most runs will be simple and edge cases can be handled later
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model_a, self.tokenizer_a, self.optimizer_a, self.lr_scheduler_a
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )

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
        args = self.args
        accelerator = self.accelerator
        optimizer_a = self.optimizer_a
        optimizer_b = self.optimizer_b
        model_a = self.model_a
        model_b = self.model_b
        tokenizer_a = self.tokenizer_a
        tokenizer_b = self.tokenizer_b
        device = accelerator.device

        self.train_dataloader_a = self.get_train_dataloader(self.train_dataset_a, self.data_collator_a)
        self.train_dataloader_b = self.get_train_dataloader(self.train_dataset_b, self.data_collator_b)

        # Trainer state initialisation
        self.state.global_step = 0
        self.state.epoch = 0
        self.state.max_steps = args.max_steps
        self.state.num_train_epochs = args.num_train_epochs

        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self.evaluate()

        for epoch in range(self.state.epoch, args.num_train_epochs):
            for idx, (batch_a, batch_b) in enumerate(zip(self.train_dataloader_a, self.train_dataloader_b)):
                # Check if training should stop
                if self.control.should_training_stop:
                    break

                # on_step_begin
                self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                # Perform cycle steps
                metrics_a = self._cycle_step(model_a, model_b, tokenizer_a, tokenizer_b, optimizer_a, batch_b, idx, cycle_name="A")
                metrics_b = self._cycle_step(model_b, model_a, tokenizer_b, tokenizer_a, optimizer_b, batch_a, idx, cycle_name="B")

                # Update state and check control flow
                self.state.global_step += 1
                self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                # Handle logging
                if self.control.should_log:
                    metrics = {**metrics_a, **metrics_b}
                    self.control = self.callback_handler.on_log(args, self.state, self.control, metrics)

                # Handle evaluation
                if self.control.should_evaluate:
                    metrics = self.evaluate()
                    self.control = self.callback_handler.on_evaluate(args, self.state, self.control, metrics)

                # Handle saving
                if self.control.should_save:
                    self.control = self.callback_handler.on_save(args, self.state, self.control)

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)

        # End of training
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, 0.0, {})

    def _cycle_step(self, model_train, model_gen, tokenizer_train, tokenizer_gen, optimizer, batch, idx, cycle_name):
        """
        Perform a single cycle step        
        """
        model_gen.eval()
        model_train.train()
        metrics = {}

        with torch.inference_mode():
            # TODO: Add generation config
            synth_batch_ids = model_gen.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_new_tokens=100)

        synth_batch = self._cycle_prepare_inputs(batch["input_ids"], synth_batch_ids, model_gen, model_train, tokenizer_gen, tokenizer_train, cycle_name)
        synth_batch = {k: v.to(self.accelerator.device) for k, v in synth_batch.items()}        

        # Accelerator fails when handling alternating gradients
        outputs = model_train(**synth_batch)
        loss = outputs.loss / self.args.gradient_accumulation_steps
        self.accelerator.backward(loss)

        if (idx+1) % self.args.gradient_accumulation_steps == 0:
            metrics[f"train_loss_{cycle_name}"] = loss.detach().float().item()
            metrics[f"learning_rate_{cycle_name}"] = optimizer.param_groups[0]["lr"]
            optimizer.step()
            optimizer.zero_grad()

        # Free up memory
        del synth_batch

        return metrics

    def _cycle_prepare_inputs(self, real_input_ids, synth_input_ids, model_gen, model_train, tokenizer_gen, tokenizer_train, cycle_name):
        """Scenarios that need testing:
        1) Order of inputs is correct
        2) No inputs have been truncated in any way
        3) -100 is set for everything except the real_input_ids
        4) Attention mask correctly accounts for padding
        """
        if not model_gen.config.is_encoder_decoder:
            synth_input_ids = synth_input_ids[:, real_input_ids.shape[-1]:]

        # TODO: Skip retokenization if tokenizers are identical
        synth_batch_text = tokenizer_gen.batch_decode(synth_input_ids, skip_special_tokens=True)

        if not model_train.config.is_encoder_decoder:
            input_texts = tokenizer_train.batch_decode(real_input_ids, skip_special_tokens=True)
            # TODO: Investigate tokenizer_train.eos_token as separator to appear more like packed training instances
            synth_batch_text = [synth_text + " " + input_text for synth_text, input_text in zip(synth_batch_text, input_texts)] # FIXME: Work out how best to separate two sequences in causal

        synth_batch = tokenizer_train(synth_batch_text, return_tensors="pt", padding=True)

        # Everything up to -real_input_ids.shape[-1] is the prompt therefore -100
        synth_batch['labels'] = synth_batch["input_ids"].clone()
        synth_batch['labels'][:, :-real_input_ids.shape[-1]] = -100 # TODO: Make sure this doesn't include leasing whitespace

        return synth_batch
    

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
        # Dataset already contains tokenized inputs
        batch = data_collator([batch])
        batch = {k: v.to(model.device) for k, v in batch.items()}
        
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

    # MODEL_NAME = "google/flan-t5-small"
    # model_a = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    # model_b = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    model_a = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)
    model_b = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)

    print('='*60)
    print(model_a.config.is_encoder_decoder)
    print(model_b.config.is_encoder_decoder)
    print('='*60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding=True)

    args = CycleTrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        logging_steps=1,
        logging_strategy="steps",
    )

    trainer = CycleTrainer(
        args,
        model_a=model_a,
        model_b=model_b,
        tokenizer_a=tokenizer,
        tokenizer_b=tokenizer,
        train_dataset_a=dataset_en['train'],
        train_dataset_b=dataset_de['train'],
        eval_dataset_a=dataset_en['test'],
        eval_dataset_b=dataset_de['test']
    )
    trainer.train()

    print("English to German")
    generate_model_samples(trainer.model_b, trainer.tokenizer_b, trainer.eval_dataset_a, trainer.data_collator_b)

    print("German to English")
    generate_model_samples(trainer.model_a, trainer.tokenizer_a, trainer.eval_dataset_b, trainer.data_collator_a)