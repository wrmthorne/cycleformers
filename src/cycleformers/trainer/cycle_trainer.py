import math
import torch
import torch.nn as nn
from transformers import DataCollatorForSeq2Seq, DataCollatorWithPadding, Trainer, PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerState, TrainerControl, PrinterCallback, CallbackHandler, ExportableState
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer_utils import TrainOutput
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from peft import PeftModel, PeftConfig
from contextlib import contextmanager

from .cycle_training_arguments import CycleTrainingArguments
from .utils import auto_temp_attributes


class CycleTrainer(Trainer):
    def __init__(
            self,
            args: CycleTrainingArguments,
            models: nn.Module | dict[str, nn.Module] | None = None,
            tokenizers: PreTrainedTokenizerBase | dict[str, PreTrainedTokenizerBase] | None = None,
            train_dataset_A = None,
            train_dataset_B = None,
            eval_dataset_A = None,
            eval_dataset_B = None,
            data_collator_A = None,
            data_collator_B = None,
            callbacks = None,
            peft_configs: PeftConfig | dict[str, PeftConfig] | None = None # TODO: Integrate later
        ):       
        self.args = args

        # Handle model initialization
        if isinstance(models, dict):
            self.model_A = models["A"]
            self.model_B = models["B"]
            self.is_peft_model = False
        elif isinstance(models, PeftModel):
            adapter_names = list(models.peft_config)
            if "A" in adapter_names and "B" in adapter_names:
                self.model_A = self.model_B = models
                self.is_peft_model = True
            else:
                raise ValueError(f"Missing at least one adapter. Expecting 'A' and 'B' but got {adapter_names}")
        else:
            raise ValueError(f"Got unexpected model type: {type(models)}")

        # Store adapter names for convenience
        self.adapter_A = "A" if self.is_peft_model else None
        self.adapter_B = "B" if self.is_peft_model else None
        
        # Extract tokenizers from input
        if isinstance(tokenizers, dict):
            tokenizer_A = tokenizers["A"] 
            tokenizer_B = tokenizers["B"]
        elif isinstance(tokenizers, PreTrainedTokenizerBase):
            tokenizer_A = tokenizer_B = tokenizers
        else:
            raise ValueError(f"Got unexpected tokenizer type: {type(tokenizers)}")

        # Configure padding for both tokenizers
        for tokenizer, model in [(tokenizer_A, self.model_A), (tokenizer_B, self.model_B)]:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.generation_config.pad_token_id = tokenizer.eos_token_id
                # TODO: Check whether seq2seq tokenizers care about padding side 
                tokenizer.padding_side = "left" if not model.config.is_encoder_decoder else tokenizer.padding_side

        self.tokenizer_A = tokenizer_A
        self.tokenizer_B = tokenizer_B

        if data_collator_A is None and self.model_A.config.is_encoder_decoder:
            self.data_collator_A = DataCollatorForSeq2Seq(tokenizer_A)
        elif data_collator_A is None:
            self.data_collator_A = DataCollatorWithPadding(tokenizer_A)

        if data_collator_B is None and self.model_B.config.is_encoder_decoder:
            self.data_collator_B = DataCollatorForSeq2Seq(tokenizer_B)
        elif data_collator_B is None:
            self.data_collator_B = DataCollatorWithPadding(tokenizer_B)

        self.train_dataset_A = train_dataset_A
        self.train_dataset_B = train_dataset_B
        self.eval_dataset_A = eval_dataset_A
        self.eval_dataset_B = eval_dataset_B

        ## Calculate batches 
        if not args.max_steps > 0:
            # Calculate number of samples per epoch
            num_samples_per_epoch = min(len(train_dataset_A), len(train_dataset_B))
            # Calculate number of steps considering batch size and gradient accumulation
            num_update_steps_per_epoch = num_samples_per_epoch // (args.per_device_train_batch_size * args.gradient_accumulation_steps)
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            args.max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
        
        accelerator = Accelerator()
        self.accelerator = accelerator

        ## Setup model and dataloaders
        # TODO: Investigate just preparing lora_layers https://github.com/huggingface/diffusers/issues/4046
        if self.is_peft_model:
            # For PEFT models, we need to set the active adapter before creating optimizers
            self.model_A.set_adapter(self.adapter_A)
        self.optimizer_A, self.lr_scheduler_A = self.create_optimizer_and_scheduler(self.model_A, args.max_steps)
            
        if self.is_peft_model:
            self.model_B.set_adapter(self.adapter_B) 
        self.optimizer_B, self.lr_scheduler_B = self.create_optimizer_and_scheduler(self.model_B, args.max_steps)


        # Prepare models and optimizers with accelerator
        prepared = self.accelerator.prepare(
            self.model_A, self.optimizer_A, self.lr_scheduler_A,
            self.model_B, self.optimizer_B, self.lr_scheduler_B
        )
        self.model_A, self.optimizer_A, self.lr_scheduler_A, self.model_B, \
            self.optimizer_B, self.lr_scheduler_B = prepared
        
        ## Trainer specific setup

        # TODO: Major revision of control flow needed for any non-trivial runs
        # Initialise callbacks and control flow - Current strategy is to just track one model
        # under the assumption that most runs will be simple and edge cases can be handled later
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model_A, self.tokenizer_A, self.optimizer_A, self.lr_scheduler_A
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
        optimizer_A = self.optimizer_A
        optimizer_B = self.optimizer_B
        scheduler_A = self.lr_scheduler_A
        scheduler_B = self.lr_scheduler_B
        model_A = self.model_A
        model_B = self.model_B
        tokenizer_A = self.tokenizer_A
        tokenizer_B = self.tokenizer_B

        self.train_dataloader_A = self.get_train_dataloader(self.train_dataset_A, self.data_collator_A)
        self.train_dataloader_B = self.get_train_dataloader(self.train_dataset_B, self.data_collator_B)

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
            for idx, (batch_A, batch_B) in enumerate(zip(self.train_dataloader_A, self.train_dataloader_B)):
                # Check if training should stop
                if self.control.should_training_stop:
                    break

                # on_step_begin
                self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                # Perform cycle steps
                metrics_A = self._cycle_step(
                    model_train=model_A,
                    model_gen=model_B,
                    tokenizer_train=tokenizer_A,
                    tokenizer_gen=tokenizer_B,
                    optimizer=optimizer_A,
                    scheduler=scheduler_A,
                    batch=batch_B,
                    idx=idx,
                    cycle_name="A"
                )
                metrics_B = self._cycle_step(
                    model_train=model_B,
                    model_gen=model_A,
                    tokenizer_train=tokenizer_B,
                    tokenizer_gen=tokenizer_A,
                    optimizer=optimizer_B,
                    scheduler=scheduler_B,
                    batch=batch_A,
                    idx=idx,
                    cycle_name="B"
                )

                del batch_A, batch_B
                torch.cuda.empty_cache()

                # Update state and check control flow
                self.state.global_step += 1
                self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                # Handle logging
                if self.control.should_log:
                    metrics = {**metrics_A, **metrics_B}
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

    @contextmanager 
    def switch_adapter(self, model: nn.Module, adapter_name: str):
        """Context manager to safely switch adapters and handle cleanup"""
        if not self.is_peft_model:
            yield
            return
            
        previous_adapter = model.active_adapter
        try:
            model.set_adapter(adapter_name)
            yield
        finally:
            model.set_adapter(previous_adapter)

    def _cycle_step(self, model_train, model_gen, tokenizer_train, tokenizer_gen, optimizer, scheduler, batch, idx, cycle_name):
        """Perform a single cycle step"""
        model_gen.eval()
        model_train.train()
        metrics = {}

        # Handle adapter switching for generation
        gen_adapter = self.adapter_B if cycle_name == "A" else self.adapter_A
        train_adapter = self.adapter_A if cycle_name == "A" else self.adapter_B

        with self.switch_adapter(model_gen, gen_adapter):
            with torch.inference_mode():
                synth_batch_ids = model_gen.generate(
                    input_ids=batch["input_ids"], 
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=100
                )

        # Prepare inputs using generated text
        synth_batch = self._cycle_prepare_inputs(
            batch["input_ids"], synth_batch_ids, model_gen, model_train, 
            tokenizer_gen, tokenizer_train, cycle_name
        )
        synth_batch = {k: v.to(self.accelerator.device) for k, v in synth_batch.items()}

        # Switch to training adapter and perform forward pass
        with self.switch_adapter(model_train, train_adapter):
            outputs = model_train(**synth_batch)
            loss = outputs.loss / self.args.gradient_accumulation_steps
            self.accelerator.backward(loss)

            # Update if needed
            if (idx+1) % self.args.gradient_accumulation_steps == 0:
                metrics[f"train_loss_{cycle_name}"] = loss.detach().float().item()
                metrics[f"learning_rate_{cycle_name}"] = optimizer.param_groups[0]["lr"]
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        # Cleanup
        del synth_batch, synth_batch_ids
        torch.cuda.empty_cache()

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
        self.model_A.eval()
        losses_A = []

        eval_dataloader_A = self.get_eval_dataloader(self.eval_dataset_A, self.data_collator_A)
        total_A = len(eval_dataloader_A)
        
        for batch in tqdm(eval_dataloader_A, total=total_A, desc="Evaluating Model A"):
            with torch.no_grad():
                outputs = self.model_A(**batch)
                loss = outputs.loss
                losses_A.append(loss.detach().cpu())
                
        metrics["eval_loss_A"] = torch.mean(torch.tensor(losses_A))

        # Evaluate model B  
        self.model_B.eval()
        losses_B = []

        eval_dataloader_B = self.get_eval_dataloader(self.eval_dataset_B, self.data_collator_B)
        total_B = len(eval_dataloader_B)

        for batch in tqdm(eval_dataloader_B, total=total_B, desc="Evaluating Model B"):
            with torch.no_grad():
                outputs = self.model_B(**batch)
                loss = outputs.loss
                losses_B.append(loss.detach().cpu())

        metrics["eval_loss_B"] = torch.mean(torch.tensor(losses_B))

        return metrics