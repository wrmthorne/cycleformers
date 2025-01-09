import math
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

import datasets
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)
from transformers.integrations import get_reporting_integration_callbacks
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
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_utils import TrainerMemoryTracker, TrainOutput

from .cycle_trainer_utils import PreTrainingSummary, load_model
from .cycles import PrepareCycleInputsNotSet, _default_prepare_cycle_inputs, _prepare_causal_skip_cycle_inputs
from .exceptions import CycleModelError, InvalidCycleKeyError, MACCTModelError, MissingModelError
from .import_utils import is_liger_kernel_available, is_peft_available
from .utils import DEFAULT_SEP_SEQ, auto_temp_attributes


if TYPE_CHECKING:
    from cycleformers import CycleTrainingArguments

if is_peft_available():
    from peft import LoraConfig, PeftConfig, PeftMixedModel, PeftModel, get_peft_model


class PeftModelProtocol(Protocol):
    """Protocol defining the required attributes and methods for PEFT models."""

    active_adapter: str
    base_model: PreTrainedModel
    config: PretrainedConfig
    generation_config: PretrainedConfig
    peft_config: dict[str, "LoraConfig"]

    def set_adapter(self, adapter_name: str) -> None: ...
    def add_adapter(self, adapter_name: str, peft_config: PeftConfig) -> None: ...
    def eval(self) -> None: ...
    def train(self) -> None: ...
    def generate(self, **kwargs) -> torch.Tensor: ...
    def __call__(self, **kwargs) -> torch.Tensor: ...


# Type variable for PEFT models, bound to our protocol
PeftModelType = TypeVar("PeftModelType", bound=PeftModelProtocol)


class CycleTrainer(Trainer):
    """A trainer class that implements cycle training for language models.

    CycleTrainer extends the Hugging Face Trainer to support cycle training between two language models.
    It can handle both encoder-decoder and causal language models, with support for PEFT adapters.

    The trainer implements a cycle where:
    1. Model A generates text from Model B's training data
    2. Model A is trained to reconstruct Model B's original input
    3. Model B generates text from Model A's training data
    4. Model B is trained to reconstruct Model A's original input

    Args:
        args (CycleTrainingArguments): Training arguments specific to cycle training
        models (nn.Module | dict[str, nn.Module] | None): The models to train. Can be:
            - A single model with two PEFT adapters named 'A' and 'B'
            - A dictionary with keys 'A' and 'B' containing separate models
            - None (for testing/subclassing)
        tokenizers (PreTrainedTokenizerBase | dict[str, PreTrainedTokenizerBase] | None):
            The tokenizers to use. Can be:
            - A single tokenizer shared between both models
            - A dictionary with keys 'A' and 'B' containing separate tokenizers
            - None (for testing/subclassing)
        train_dataset_A: Training dataset for model A
        train_dataset_B: Training dataset for model B
        eval_dataset_A: Evaluation dataset for model A
        eval_dataset_B: Evaluation dataset for model B
        data_collator_A: Data collator for model A
        data_collator_B: Data collator for model B
        callbacks: List of callbacks to use during training
        peft_configs (PeftConfig | dict[str, PeftConfig] | None): PEFT configurations. Can be:
            - A single config to create identical adapters
            - A dictionary with keys 'A' and 'B' for different configurations
            - None if not using PEFT

    Examples:
        Basic usage with two separate models:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from cycleformers import CycleTrainingArguments
        >>>
        >>> # Initialize models and tokenizers
        >>> model_A = AutoModelForCausalLM.from_pretrained("gpt2-small")
        >>> model_B = AutoModelForCausalLM.from_pretrained("gpt2-small")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2-small")
        >>> tokenizer.pad_token = tokenizer.eos_token
        >>>
        >>> # Create trainer
        >>> trainer = CycleTrainer(
        ...     args=CycleTrainingArguments(output_dir="./output"),
        ...     models={"A": model_A, "B": model_B},
        ...     tokenizers=tokenizer,
        ...     train_dataset_A=train_dataset_A,
        ...     train_dataset_B=train_dataset_B
        ... )

        Using PEFT with a single model and two adapters:
        >>> from peft import LoraConfig
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from cycleformers import CycleTrainingArguments
        >>>
        >>> # Initialize base model and tokenizer
        >>> base_model = AutoModelForCausalLM.from_pretrained("gpt2-small")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2-small")
        >>> tokenizer.pad_token = tokenizer.eos_token
        >>>
        >>> # Create PEFT config
        >>> peft_config = LoraConfig(
        ...     r=8,
        ...     lora_alpha=32,
        ...     target_modules=["q_proj", "v_proj"],
        ...     lora_dropout=0.05,
        ...     bias="none"
        ... )
        >>>
        >>> # Create trainer with PEFT
        >>> trainer = CycleTrainer(
        ...     args=CycleTrainingArguments(output_dir="./output"),
        ...     models=base_model,
        ...     tokenizers=tokenizer,
        ...     train_dataset_A=train_dataset_A,
        ...     train_dataset_B=train_dataset_B,
        ...     peft_configs=peft_config  # Will create two identical adapters
        ... )

        Using different PEFT configs for each adapter:
        >>> peft_configs = {
        ...     "A": LoraConfig(r=8, lora_alpha=32),
        ...     "B": LoraConfig(r=16, lora_alpha=64)
        ... }
        >>> trainer = CycleTrainer(
        ...     args=CycleTrainingArguments(output_dir="./output"),
        ...     models=base_model,
        ...     tokenizers=tokenizer,
        ...     train_dataset_A=train_dataset_A,
        ...     train_dataset_B=train_dataset_B,
        ...     peft_configs=peft_configs  # Different config for each adapter
        ... )
    """

    def __init__(
        self,
        args: "CycleTrainingArguments",
        models: nn.Module | dict[str, nn.Module] | str | PathLike[str] | None = None,
        tokenizers: PreTrainedTokenizerBase | dict[str, PreTrainedTokenizerBase] | str | PathLike[str] | None = None,
        train_dataset_A: datasets.Dataset | None = None,
        train_dataset_B: datasets.Dataset | None = None,
        eval_dataset_A: datasets.Dataset | None = None,
        eval_dataset_B: datasets.Dataset | None = None,
        data_collator_A: DataCollatorWithPadding | DataCollatorForSeq2Seq | None = None,
        data_collator_B: DataCollatorWithPadding | DataCollatorForSeq2Seq | None = None,
        callbacks: list[TrainerCallback] | None = None,
        peft_configs: "PeftConfig | dict[str, PeftConfig] | None" = None,
    ):
        self.args = args
        self.is_macct_model = self.args.use_macct

        self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        self._memory_tracker.start()

        # Models is None or empty dict
        if models is None or not models:
            raise MissingModelError("CycleTrainer didn't receive any models or paths to train.")

        # Validate peft_configs and convert to dict for easier handling
        if is_peft_available() and peft_configs is not None:
            # Single config for both models
            if isinstance(peft_configs, PeftConfig):
                peft_configs = {"A": peft_configs, "B": peft_configs}
            # Allow A or B but must have at least one
            elif isinstance(peft_configs, dict):
                if not any(key in peft_configs.keys() for key in ["A", "B"]):
                    raise InvalidCycleKeyError("peft_configs dict must contain at least one of the keys 'A' or 'B'")
            elif not isinstance(peft_configs, (PeftConfig, dict)):
                raise ValueError(
                    f"peft_configs must be a PeftConfig or dict[str, PeftConfig], got {type(peft_configs)}"
                )

        # TODO: Should strings only be able to create MACCT, only multi-model or both?
        # TODO: Be consistent across all single model cases
        if isinstance(models, (str, PathLike)):
            # Load model from path
            model_name = models
            model_A = load_model(model_name, **self.args.model_init_kwargs)

            if not self.is_macct_model:
                # Create a duplicate model for non-MACCT mode
                model_B = load_model(model_name, **self.args.model_init_kwargs)
                models = {"A": model_A, "B": model_B}
            else:
                models = model_A

        # === Single Models === #
        if isinstance(models, nn.Module):
            if not self.is_macct_model:
                raise MissingModelError()

            if not is_peft_available():
                raise MACCTModelError(
                    "PEFT is not available. Multi-adapter training cannot be performed without "
                    "PEFT. Please install it with `pip install peft`"
                )

            # No way to create adapters without a PeftConfig
            if not isinstance(models, (PeftModel, PeftMixedModel)):
                if peft_configs is None or not all(key in peft_configs for key in ["A", "B"]):
                    raise MACCTModelError(
                        "MACCT mode requires PEFT adapters. Please provide peft_configs or a PeftModel "
                        "with adapters 'A' and 'B'."
                    )
                models = get_peft_model(models, peft_configs["A"], adapter_name="A")
                models.add_adapter("B", peft_configs["B"])
            else:
                # Check for required adapters
                missing_adapters = {"A", "B"} - set(models.peft_config)
                if missing_adapters:
                    if peft_configs is None:
                        raise MACCTModelError(
                            f"PeftModel is missing adapter(s) {missing_adapters} and no PeftConfig was provided. "
                            "Please provide peft_configs or add the missing adapters."
                        )
                    # Add missing adapters from configs
                    for adapter in missing_adapters:
                        if adapter not in peft_configs:
                            raise MACCTModelError(
                                f"Missing config for adapter {adapter} in peft_configs. Please provide "
                                "configs for all missing adapters."
                            )
                        models.add_adapter(adapter, peft_configs[adapter])
                        warnings.warn(
                            f"Adding adapter {adapter} from peft_configs. This is not expected if you are "
                            "loading pre-trained adapters `A` and `B`."
                        )

            # TODO: Replace with single model management object (e.g. ModelManager or PeftModel)
            # Use same model for both A and B in MACCT mode
            if not isinstance(models, (PeftModel, PeftMixedModel)):
                raise MACCTModelError("Model must be a PeftModel or PeftMixedModel in MACCT mode")
            self.model_A = self.model_B = models

        # Handle dictionary of models
        elif isinstance(models, dict):
            if self.is_macct_model:
                raise MACCTModelError("Cannot use dictionary of models in MACCT mode. Please provide a single model.")
            if not models.keys() == {"A", "B"}:
                raise InvalidCycleKeyError(
                    f"models dict must contain exact keys 'A' and 'B'. Got {list(models.keys())}"
                )

            if is_peft_available() and peft_configs is not None:
                # Apply PEFT configs to models if they are supplied and not already present
                for key in ["A", "B"]:
                    if not isinstance(models[key], PeftModel) or models[key].peft_config.get(key) is None:
                        if isinstance(peft_configs, dict):
                            peft_config_key = peft_configs.get(key, None)
                        else:
                            peft_config_key = peft_configs

                        if peft_config_key is not None:
                            models[key] = get_peft_model(models[key], peft_config_key, adapter_name=key)

            self.model_A = cast(PeftModel | PeftMixedModel, models["A"])
            self.model_B = cast(PeftModel | PeftMixedModel, models["B"])

        else:
            raise CycleModelError(
                f"Invalid models type: {type(models)}. Must be one of: PreTrainedModel, PeftModel, "
                "dict[str, nn.Module], or path to model."
            )

        self.adapter_A = "A" if self.is_macct_model else None
        self.adapter_B = "B" if self.is_macct_model else None

        if self.args.use_liger_kernel:
            if is_liger_kernel_available():
                from liger_kernel.transformers import _apply_liger_kernel_to_instance  # type: ignore

                for model in [self.model_A] + [self.model_B] if not self.is_macct_model else []:
                    if isinstance(model, PreTrainedModel):
                        # Patch the model with liger kernels. Use the default kernel configurations.
                        _apply_liger_kernel_to_instance(model=model)
                    elif hasattr(model, "base_model") and isinstance(model.base_model.model, PreTrainedModel):
                        # Patch the base model with liger kernels. Use the default kernel configurations.
                        _apply_liger_kernel_to_instance(model=model.base_model.model)
                    else:
                        warnings.warn(
                            "The model is not an instance of PreTrainedModel. No liger kernels will be applied."
                        )
            else:
                raise ImportError(
                    "You have set use_liger_kernel to True but liger-kernel >= 0.3.0 is not available. "
                    "Please install it with pip install liger-kernel"
                )

        # Create distinct tokenizers if not provided
        if tokenizers is None:
            tokenizer_A = AutoTokenizer.from_pretrained(self._get_config(self.model_A).name_or_path)
            tokenizer_B = AutoTokenizer.from_pretrained(self._get_config(self.model_B).name_or_path)
            tokenizers = {"A": tokenizer_A, "B": tokenizer_B}

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
            tokenizer.padding_side = (
                "left" if not self._get_config(model).is_encoder_decoder else tokenizer.padding_side
            )

        self.tokenizer_A = tokenizer_A
        self.tokenizer_B = tokenizer_B

        if data_collator_A is None:
            if self._get_config(self.model_A).is_encoder_decoder:
                self.data_collator_A = DataCollatorForSeq2Seq(tokenizer_A)
            else:
                self.data_collator_A = DataCollatorWithPadding(tokenizer_A)

        if data_collator_B is None:
            if self._get_config(self.model_B).is_encoder_decoder:
                self.data_collator_B = DataCollatorForSeq2Seq(tokenizer_B)
            else:
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
                    is_encoder_decoder=self._get_config(model).is_encoder_decoder,
                    formatting_func=None,
                    add_special_tokens=True,
                    skip_prepare_dataset=False,
                )
                # Only hack I could find to not have lots of repeat code
                setattr(self, attr_name, dataset)

        # Calculate batches
        if not args.max_steps > 0:
            if train_dataset_A is None or train_dataset_B is None:
                raise ValueError("Both train_dataset_A and train_dataset_B must be provided")

            num_samples_per_epoch = min(len(train_dataset_A), len(train_dataset_B))
            num_update_steps_per_epoch = num_samples_per_epoch // (
                args.per_device_train_batch_size * args.gradient_accumulation_steps
            )
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            args.max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)

        accelerator = Accelerator()
        self.accelerator = accelerator

        ## Setup model and dataloaders
        # TODO: Investigate just preparing lora_layers https://github.com/huggingface/diffusers/issues/4046
        if self.is_macct_model and self.adapter_A is not None:
            # For PEFT models, we need to set the active adapter before creating optimizers
            self.model_A.set_adapter(self.adapter_A)
        self.optimizer_A, self.lr_scheduler_A = self.create_optimizer_and_scheduler(self.model_A, args.max_steps)

        if self.is_macct_model and self.adapter_B is not None:
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
        self._memory_tracker.stop_and_update_metrics()

    def _get_config(self, model: "PreTrainedModel | PeftModel | PeftMixedModel") -> PretrainedConfig:
        if isinstance(model, (PeftModel, PeftMixedModel)):
            return model.base_model.config
        return model.config

    def _prepare_dataset(
        self,
        dataset: datasets.Dataset,
        processing_class: PreTrainedTokenizerBase,
        text_column: str,
        max_seq_length: int | None,
        remove_unused_columns: bool,
        is_encoder_decoder: bool,
        formatting_func: Callable | None = None,
        add_special_tokens: bool = True,
        skip_prepare_dataset: bool = False,
    ) -> datasets.Dataset:
        """Prepares a dataset for training by tokenizing text and formatting inputs.

        This method processes raw text datasets into tokenized format suitable for training.
        It handles both encoder-decoder and causal language models, with support for custom
        text formatting.

        Args:
            dataset (Dataset): HuggingFace dataset to process
            processing_class (PreTrainedTokenizerBase): Tokenizer to use for processing
            text_column (str): Name of column containing text data
            max_seq_length (int | None): Maximum sequence length for tokenization
            remove_unused_columns (bool): Whether to remove columns not used in training
            is_encoder_decoder (bool): Whether the model is an encoder-decoder
            formatting_func (Callable | None): Optional function to format text before tokenization.
                Must return a list of strings. Defaults to None.
            add_special_tokens (bool): Whether to add special tokens during tokenization.
                Defaults to True.
            skip_prepare_dataset (bool): Skip processing and return dataset as-is.
                Defaults to False.

        Returns:
            Dataset: Processed dataset containing 'input_ids' and 'attention_mask'

        Raises:
            ValueError: If dataset is None or formatting_func doesn't return a list

        Examples:
            Basic usage:
            >>> from datasets import Dataset
            >>> dataset = Dataset.from_dict({"text": ["Hello world", "How are you?"]})
            >>> processed = trainer._prepare_dataset(
            ...     dataset=dataset,
            ...     processing_class=tokenizer,
            ...     text_column="text",
            ...     max_seq_length=128,
            ...     remove_unused_columns=True,
            ...     is_encoder_decoder=False
            ... )
            >>> processed[0].keys()
            dict_keys(['input_ids', 'attention_mask'])

            With custom formatting:
            >>> def format_text(example):
            ...     return [f"Question: {text}" for text in example["text"]]
            >>> processed = trainer._prepare_dataset(
            ...     dataset=dataset,
            ...     processing_class=tokenizer,
            ...     text_column="text",
            ...     max_seq_length=128,
            ...     remove_unused_columns=True,
            ...     is_encoder_decoder=False,
            ...     formatting_func=format_text
            ... )
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
        """Get a dataloader for training"""
        dataloader_params = {"batch_size": self.args.per_device_train_batch_size, "collate_fn": data_collator}
        return self.accelerator.prepare(DataLoader(dataset, **dataloader_params))

    def get_eval_dataloader(self, dataset, data_collator):
        """Get a dataloader for evaluation"""
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

        if self.is_macct_model:
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

    def train(self) -> TrainOutput:
        """Train models using cycle training.

        Returns:
            TrainOutput: Contains training metrics and state

        Examples:
            >>> trainer.train()
            TrainOutput(global_step=1000, training_loss=2.4, metrics={})
        """
        # Must start as early as possible
        self._memory_tracker.start()

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

        # TODO: Tidy this up
        PreTrainingSummary(
            {"A": self.model_A, "B": self.model_B},
            {"A": self._get_config(self.model_A), "B": self._get_config(self.model_B)},
            {"A_train": self.train_dataset_A, "B_train": self.train_dataset_B},
            self.is_macct_model,
        )
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
        self._memory_tracker.stop_and_update_metrics(metrics)
        self.log(metrics)

        return TrainOutput(self.state.global_step, 0.0, {})

    @contextmanager
    def switch_adapter(self, model: "PeftModel | PeftMixedModel | nn.Module", adapter_name: str | None = None):
        """Context manager to safely switch adapters and handle cleanup"""
        if not self.is_macct_model or adapter_name is None or not isinstance(model, (PeftModel, PeftMixedModel)):
            yield
            return

        previous_adapter = model.active_adapter
        try:
            # We know adapter_name is str here because we checked for None above
            adapter_str: str = adapter_name
            if isinstance(model, PeftModel):
                model.set_adapter(adapter_str)
            elif isinstance(model, PeftMixedModel):
                model.set_adapter(adapter_str)  # PeftMixedModel accepts str or list[str]
            yield
        finally:
            model.set_adapter(previous_adapter)

    def _cycle_step(
        self,
        model_train: "PeftModel | PeftMixedModel | nn.Module",
        model_gen: "PeftModel | PeftMixedModel | nn.Module",
        tokenizer_train: PreTrainedTokenizerBase,
        tokenizer_gen: PreTrainedTokenizerBase,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        batch: dict,
        idx: int,
        cycle_name: str,
    ) -> dict:
        """Perform a single training step in the cycle training process.

        This method handles the core cycle training logic for a single step:
        1. Generate synthetic text using model_gen
        2. Process the synthetic text into training inputs
        3. Train model_train to reconstruct the original input

        Args:
            model_train (nn.Module): The model being trained in this step
            model_gen (nn.Module): The model generating synthetic text
            tokenizer_train (PreTrainedTokenizerBase): Tokenizer for model_train
            tokenizer_gen (PreTrainedTokenizerBase): Tokenizer for model_gen
            optimizer (torch.optim.Optimizer): Optimizer for model_train
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
            batch (dict): Current training batch containing input_ids and attention_mask
            idx (int): Current batch index
            cycle_name (str): Name of current cycle ("A" or "B")

        Returns:
            dict: Training metrics for this step including:
                - train_loss_{cycle_name}: Training loss for this step
                - learning_rate_{cycle_name}: Current learning rate

        Examples:
            >>> metrics = trainer._cycle_step(
            ...     model_train=model_A,
            ...     model_gen=model_B,
            ...     tokenizer_train=tokenizer_A,
            ...     tokenizer_gen=tokenizer_B,
            ...     optimizer=optimizer_A,
            ...     scheduler=scheduler_A,
            ...     batch={"input_ids": ids, "attention_mask": mask},
            ...     idx=0,
            ...     cycle_name="A"
            ... )
            >>> metrics
            {'train_loss_A': 2.4, 'learning_rate_A': 5e-5}
        """
        model_gen.eval()
        model_train.train()
        metrics = {}

        TEMP_GEN_ARGS = {
            "max_new_tokens": 30,
            "use_cache": True,
            "do_sample": False,  # Significant speedup
        }

        # Handle adapter switching for generation
        gen_adapter = self.adapter_B if cycle_name == "A" else self.adapter_A
        train_adapter = self.adapter_A if cycle_name == "A" else self.adapter_B

        with self.switch_adapter(model_gen, gen_adapter):
            with torch.inference_mode():
                synth_batch_ids = model_gen.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **TEMP_GEN_ARGS,
                    pad_token_id=tokenizer_gen.eos_token_id if tokenizer_gen.eos_token_id else None,
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
        """Set the function used to prepare inputs during cycle training.

        This method controls how generated text is processed into training inputs for the next model.
        If no function is provided, an optimized implementation is automatically selected based on the
        model types and tokenizers.

        Args:
            fn (Callable | None): Function to prepare cycle inputs. Should accept:
                - real_input_ids: Original input token IDs
                - synth_input_ids: Generated token IDs
                - model_gen: The model that generated the text
                - model_train: The model being trained
                - tokenizer_gen: Tokenizer for the generative model
                - tokenizer_train: Tokenizer for the training model
                - cycle_name: Name of current cycle ("A" or "B")

        Examples:
            >>> def custom_prepare_fn(self, real_ids, synth_ids, *args, **kwargs):
            ...     # Custom implementation
            ...     return {"input_ids": ids, "attention_mask": mask, "labels": labels}
            >>>
            >>> trainer.set_cycle_inputs_fn(custom_prepare_fn)

            Using default optimization:
            >>> trainer.set_cycle_inputs_fn()  # Automatically selects best implementation
        """
        try:
            # User provided method
            if fn is not None:
                bound_method = MethodType(fn, self)
            # TODO: Add better tokenizer equality check
            # Both causal models that share a tokenizer from a tested model family
            elif (
                self.tokenizer_A == self.tokenizer_B
                and not self._get_config(self.model_A).is_encoder_decoder
                and not self._get_config(self.model_B).is_encoder_decoder
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
    ) -> dict[str, torch.Tensor]:
        """Endpoint for handling of mid-cycle token processing.

        Handle the outputs of the generative model ready for the training model. In the case of seq2seq, simply
        use the real input ids as labels and the synth input ids as input. For causal, we need to split the prompt
        from the response, clean the prompt, swap the order, add any separator tokens we want and then right pad.

        Subclass and override this method to implement custom logic. The method should return a BatchEncoding object
        or equivalent dict with input_ids, attention_mask and labels.

        For more details see the `_prepare_cycle_inputs` method in the `cycles/cycles.py` file.
        """
        raise PrepareCycleInputsNotSet()

    def evaluate(self, ignore_keys=None) -> dict[str, torch.Tensor]:
        """Evaluate the models during training.

        This method evaluates the models during training to monitor progress and adjust training parameters.
        It calculates the loss for both models and returns a dictionary with the evaluation metrics.

        Args:
            ignore_keys (list[str] | None): List of keys to ignore in the evaluation metrics

        Returns:
            dict[str, torch.Tensor]: Dictionary containing evaluation metrics for both models

        Examples:
            >>> trainer = CycleTrainer(args, models, tokenizers)
            >>> metrics = trainer.evaluate()
            >>> print(metrics)
            {'eval_loss_A': 2.1, 'eval_loss_B': 1.9}

            Ignore specific metrics:
            >>> metrics = trainer.evaluate(ignore_keys=['eval_loss_B'])
            >>> print(metrics)
            {'eval_loss_A': 2.1}
        """
        # Must start as early as possible
        self._memory_tracker.start()

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
        self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics
