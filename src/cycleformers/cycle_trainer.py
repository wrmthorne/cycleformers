import math
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import datasets
import torch
import torch.nn as nn
from accelerate import Accelerator
from peft import PeftConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, DataCollatorWithPadding, PreTrainedTokenizerBase, Trainer
from transformers.integrations import get_reporting_integration_callbacks
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer import (
    DEFAULT_CALLBACKS,
    DEFAULT_PROGRESS_CALLBACK,
    PREFIX_CHECKPOINT_DIR,
    TRAINER_STATE_NAME,
)
from transformers.trainer_callback import (
    CallbackHandler,
    ExportableState,
    PrinterCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_utils import TrainOutput

from .cycle_training_arguments import CycleTrainingArguments
from .cycles import PrepareCycleInputsNotSet, _default_prepare_cycle_inputs, _prepare_causal_skip_cycle_inputs
from .utils import DEFAULT_SEP_SEQ, auto_temp_attributes


class CycleTrainer(Trainer):
    def __init__(
        self,
        args: CycleTrainingArguments,
        models: nn.Module | dict[str, nn.Module] | None = None,
        tokenizers: PreTrainedTokenizerBase | dict[str, PreTrainedTokenizerBase] | None = None,
        train_dataset_A=None,
        train_dataset_B=None,
        eval_dataset_A=None,
        eval_dataset_B=None,
        data_collator_A=None,
        data_collator_B=None,
        callbacks=None,
        peft_configs: PeftConfig | dict[str, PeftConfig] | None = None,
    ):
        self.args = args

        # Handle model initialization
        if isinstance(models, dict):
            if peft_configs is not None:
                raise ValueError("When using peft_configs, models should be a single model, not a dictionary")
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
        elif not isinstance(models, nn.Module):
            raise ValueError("models must be a nn.Module or dict[str, nn.Module] with keys 'A' and 'B'")
        elif peft_configs is not None:
            if isinstance(peft_configs, PeftConfig):
                # Single config - create two identical adapters
                if hasattr(peft_configs, "adapter_name") and peft_configs.adapter_name is not None:
                    warnings.warn("adapter_name in peft_config will be ignored. Using 'A' and 'B' as adapter names")

                # Create model with first adapter
                self.model_A = self.model_B = get_peft_model(models, peft_configs, adapter_name="A")
                self.model_A.add_adapter("B", peft_configs)
                self.is_peft_model = True

            elif isinstance(peft_configs, dict):
                # Dictionary of configs
                if not {"A", "B"}.issubset(peft_configs.keys()):
                    raise ValueError(
                        f"peft_configs dictionary must contain at least keys 'A' and 'B'. Got {list(peft_configs)}"
                    )

                # Check for different task types
                task_types = {name: config.task_type for name, config in peft_configs.items()}
                if len(set(task_types.values())) > 1:
                    warnings.warn(
                        f"Different task types detected in peft_configs: {task_types}. "
                        "This may lead to unexpected behavior."
                    )

                # Create model with first adapter
                self.model_A = self.model_B = get_peft_model(models, peft_configs["A"], adapter_name="A")
                # Add remaining adapters
                for name, config in peft_configs.items():
                    if name != "A":
                        self.model_A.add_adapter(name, config)
                self.is_peft_model = True
            else:
                raise ValueError(
                    f"peft_configs must be a PeftConfig or dict[str, PeftConfig], got {type(peft_configs)}"
                )
        else:
            # TODO: Should this be allowed?
            self.model_A = self.model_B = models
            self.is_peft_model = False

        # Store adapter names for convenience
        self.adapter_A = "A" if self.is_peft_model else None
        self.adapter_B = "B" if self.is_peft_model else None

        # Extract tokenizers from input
        if isinstance(tokenizers, dict):
            if tokenizers.keys() != {"A", "B"}:
                raise ValueError(f"Got unexpected tokenizer keys: {tokenizers.keys()}")

            if tokenizers["A"] == tokenizers["B"]:
                tokenizer_A = tokenizer_B = tokenizers["A"]  # TODO: Find equality function for tokenizers
            else:
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
            # Set the default padding side to left so it is correct for _prepare_dataset and data collators
            # Call padding_side in tokenizer when we create final synth batches
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

        # TODO: Expose through config
        self.sep_seq = DEFAULT_SEP_SEQ

        # TODO: Separate our train and eval if we support multiple eval datasets
        for dataset, tokenizer, model, attr_name in [
            (train_dataset_A, self.tokenizer_A, self.model_A, "train_dataset_A"),
            (train_dataset_B, self.tokenizer_B, self.model_B, "train_dataset_B"),
            (eval_dataset_A, self.tokenizer_A, self.model_A, "eval_dataset_A"),
            (eval_dataset_B, self.tokenizer_B, self.model_B, "eval_dataset_B"),
        ]:
            # FIXME: Replace hard coded values once dataset configs are working
            if dataset is not None:
                dataset = self._prepare_dataset(
                    dataset=dataset,
                    processing_class=tokenizer,
                    text_column="text",
                    max_seq_length=None,
                    remove_unused_columns=True,
                    is_encoder_decoder=model.config.is_encoder_decoder,
                    formatting_func=None,
                    add_special_tokens=True,
                    skip_prepare_dataset=False,
                )
                # Only hack I could find to not have lots of repeat code
                setattr(self, attr_name, dataset)

        ## Calculate batches
        if not args.max_steps > 0:
            # Calculate number of samples per epoch
            num_samples_per_epoch = min(len(train_dataset_A), len(train_dataset_B))
            # Calculate number of steps considering batch size and gradient accumulation
            num_update_steps_per_epoch = num_samples_per_epoch // (
                args.per_device_train_batch_size * args.gradient_accumulation_steps
            )
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
            self.model_A, self.optimizer_A, self.lr_scheduler_A, self.model_B, self.optimizer_B, self.lr_scheduler_B
        )
        self.model_A, self.optimizer_A, self.lr_scheduler_A, self.model_B, self.optimizer_B, self.lr_scheduler_B = (
            prepared
        )

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

        self.set_cycle_inputs_fn()

    def _prepare_dataset(
        self,
        dataset,
        processing_class,
        text_column,
        max_seq_length,
        remove_unused_columns,
        is_encoder_decoder,
        formatting_func: Callable | None = None,
        add_special_tokens=True,
        skip_prepare_dataset=False,
    ):
        """
        Modification of TRL SFTTrainer._prepare_dataset
        https://github.com/huggingface/trl/blob/b02189aaa538f3a95f6abb0ab46c0a971bfde57e/trl/trainer/sft_trainer.py#L329
        """
        if dataset is None:
            raise ValueError("Dataset cannot be None")

        if skip_prepare_dataset:
            return dataset

        column_names = (
            dataset.column_names if isinstance(dataset, (datasets.Dataset, datasets.IterableDataset)) else None
        )

        if column_names and "input_ids" in column_names:
            if formatting_func is not None:
                warnings.warn(
                    "You passed a dataset that is already processed (contains an `input_ids` field) together with a "
                    "valid formatting function. Therefore `formatting_func` will be ignored. Either remove the "
                    "`formatting_func` or pass a dataset that is not already processed.",
                    UserWarning,
                )

            def formatting_func(x):
                return x["input_ids"]

        if isinstance(dataset, (torch.utils.data.IterableDataset, torch.utils.data.Dataset)) and not isinstance(
            dataset, datasets.IterableDataset
        ):
            return dataset

        def tokenize(element):
            if formatting_func is None and not is_encoder_decoder:
                texts = [text + self.sep_seq for text in element[text_column]]
            elif formatting_func is None and is_encoder_decoder:
                texts = element[text_column]
            else:
                texts = formatting_func(element)

            outputs = processing_class(
                texts,
                add_special_tokens=add_special_tokens,
                truncation=True,
                padding=False,
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )

            if formatting_func is not None and not isinstance(formatting_func(element), list):
                raise ValueError(
                    "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                )
            del texts
            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

        signature_columns = ["input_ids", "labels", "attention_mask"]

        if dataset.column_names is not None:  # None for IterableDataset
            extra_columns = list(set(dataset.column_names) - set(signature_columns))
        else:
            extra_columns = []

        if not remove_unused_columns and len(extra_columns) > 0:
            warnings.warn(
                "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with "
                "the default collator and yield to errors. If you want to inspect dataset other columns (in this "
                f"case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the "
                "default collator and create your own data collator in order to inspect the unused dataset columns.",
                UserWarning,
            )

        map_kwargs = {
            "batched": True,
            "remove_columns": dataset.column_names if remove_unused_columns else None,
            "batch_size": 4,  # FIXME: self.dataset_batch_size,
        }

        if isinstance(dataset, datasets.Dataset):
            map_kwargs["num_proc"] = 1  # FIXME: self.dataset_num_proc  # this arg is not available for IterableDataset
        tokenized_dataset = dataset.map(tokenize, **map_kwargs)

        return tokenized_dataset

    def get_train_dataloader(self, dataset, data_collator):
        dataloader_params = {"batch_size": self.args.per_device_train_batch_size, "collate_fn": data_collator}
        return self.accelerator.prepare(DataLoader(dataset, **dataloader_params))

    def get_eval_dataloader(self, dataset, data_collator):
        dataloader_params = {"batch_size": self.args.per_device_eval_batch_size, "collate_fn": data_collator}
        return self.accelerator.prepare(DataLoader(dataset, **dataloader_params))

    def _save_checkpoint(self, model, trial=None, metrics=None):
        """Minor reimplementation of parent class method"""
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        # Creating class variables needed to hack the trainer into saving for now
        self.hp_search_backend = None
        self.is_fsdp_enabled = False
        self.is_deepspeed_enabled = False
        self.current_flos = 0  # FIXME: Remember to remove this
        # TODO: Handle hyper-parameter searches?
        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = Path(run_dir) / checkpoint_folder

        if self.is_peft_model:
            self.save_model(output_dir, self.model_A, _internal_call=True)
        else:
            self.save_model(output_dir / "A", self.model_A, _internal_call=True)
            self.save_model(output_dir / "B", self.model_B, _internal_call=True)

        # Manually handle tokenizers
        if self.tokenizer_A != self.tokenizer_B:
            self.tokenizer_A.save_pretrained(output_dir / "A")
            self.tokenizer_B.save_pretrained(output_dir / "B")
        else:
            self.tokenizer_A.save_pretrained(output_dir)

        if not self.args.save_only_model:
            self._save_optimizer_and_scheduler(output_dir / "A", self.optimizer_A, self.lr_scheduler_A)
            self._save_optimizer_and_scheduler(output_dir / "B", self.optimizer_B, self.lr_scheduler_B)
            self._save_rng_state(output_dir)

        # TODO: Handle determination of best metrics and model checkpoint (for each model or across the cycle?)

        if self.args.should_save:
            for cb in [
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]:
                cb_name = cb.__class__.__name__
                cb_state = cb.state()
                if isinstance(self.state.stateful_callbacks[cb_name], list):
                    self.state.stateful_callbacks[cb_name].append(cb_state)
                else:
                    self.state.stateful_callbacks[cb_name] = cb_state
            self.state.save_to_json(output_dir / TRAINER_STATE_NAME)

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            # Solely rely on numerical checkpoint id for rotation.
            # mtime is not reliable especially on some fuse fs in cloud environments.
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

    @auto_temp_attributes("model", "processing_class")
    def save_model(self, output_dir: str, model, _internal_call: bool = False):
        super().save_model(output_dir, _internal_call)

    @auto_temp_attributes("optimizer", "lr_scheduler")
    def _save_optimizer_and_scheduler(self, output_dir: str, optimizer, lr_scheduler):
        super()._save_optimizer_and_scheduler(output_dir)

    @auto_temp_attributes("model", "optimizer", "optimizer_cls_and_kwargs", "lr_scheduler")
    def create_optimizer_and_scheduler(self, model, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps=num_training_steps)
        return self.optimizer, self.lr_scheduler

    def train(self):
        args = self.args
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
        epochs_trained = 0
        num_train_epochs = math.ceil(args.num_train_epochs)

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

        for epoch in range(epochs_trained, num_train_epochs):
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
                    cycle_name="A",
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
                    cycle_name="B",
                )

                del batch_A, batch_B
                torch.cuda.empty_cache()

                # Update state and check control flow
                self.state.global_step += 1

                # Add save strategy handling here
                if self.args.save_strategy == "steps":
                    if self.state.global_step % self.state.save_steps == 0:
                        self.control.should_save = True
                elif self.args.save_strategy == "epoch":
                    # Save at the end of epoch
                    if idx == len(self.train_dataloader_A) - 1:
                        self.control.should_save = True

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
                    self._save_checkpoint(self.model_A)  # We only need to pass one model
                    self.control.should_save = False  # Reset the flag after saving

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

    def _cycle_step(
        self, model_train, model_gen, tokenizer_train, tokenizer_gen, optimizer, scheduler, batch, idx, cycle_name
    ):
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
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_new_tokens=100
                )

        # Prepare inputs using generated text
        synth_batch = self._prepare_cycle_inputs(
            batch["input_ids"], synth_batch_ids, model_gen, model_train, tokenizer_gen, tokenizer_train, cycle_name
        )
        synth_batch = {k: v.to(self.accelerator.device) for k, v in synth_batch.items()}

        # Switch to training adapter and perform forward pass
        with self.switch_adapter(model_train, train_adapter):
            outputs = model_train(**synth_batch)
            loss = outputs.loss / self.args.gradient_accumulation_steps
            self.accelerator.backward(loss)

            # Update if needed
            if (idx + 1) % self.args.gradient_accumulation_steps == 0:
                metrics[f"train_loss_{cycle_name}"] = loss.detach().float().cpu().item()
                metrics[f"learning_rate_{cycle_name}"] = optimizer.param_groups[0]["lr"]
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        # Cleanup
        del synth_batch, synth_batch_ids, loss, outputs
        torch.cuda.empty_cache()

        return metrics

    def set_cycle_inputs_fn(self, fn: Callable | None = None):
        """
        Setter method to control which mid cycle token preparation method to use from cycleformers.cycles

        Method is called during class init but may be called by user post instantiation to force specific behaviour
        without subclassing. If no arg is passed, this method will automatically select the best optimised method
        for the given conditions.
        """
        try:
            # User provided method
            if fn is not None:
                bound_method = MethodType(fn, self)
            # TODO: Add better tokenizer equality check
            # Both causal models that share a tokenizer from a tested model family
            elif (
                self.tokenizer_A == self.tokenizer_B
                and not self.model_A.config.is_encoder_decoder
                and not self.model_B.config.is_encoder_decoder
            ):
                bound_method = MethodType(_prepare_causal_skip_cycle_inputs, self)
            else:
                bound_method = MethodType(_default_prepare_cycle_inputs, self)

            setattr(self, "_prepare_cycle_inputs", bound_method)
        except Exception as e:
            raise PrepareCycleInputsNotSet(f"Failed to set prepare_cycle_inputs: {e}")

    def _prepare_cycle_inputs(
        self,
        real_input_ids: torch.Tensor,
        synth_input_ids: torch.Tensor,
        model_gen: nn.Module,
        model_train: nn.Module,
        tokenizer_gen: PreTrainedTokenizerBase,
        tokenizer_train: PreTrainedTokenizerBase,
        cycle_name: str,
    ) -> BatchEncoding:
        """
        Handlle the outputs of the generative model ready for the training model. In the case of seq2seq, simply
        use the real input ids as labels and the synth input ids as input. For causal, we need to split the prompt
        from the response, clean the prompt, swap the order, add any separator tokens we want and then right pad.

        Subclass and override this method to implement custom logic. The method should return a BatchEncoding object
        or equivalent dict with input_ids, attention_mask and labels.
        """
        raise PrepareCycleInputsNotSet()

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
