import gc
from dataclasses import dataclass
from os import PathLike
from typing import Any

import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from cycleformers.exceptions import CycleModelError


@dataclass
class EvalGeneration:
    """Class for storing evaluation generation results."""

    predictions: list[str]
    labels: list[str]


def load_model(model_path: str | PathLike[str], **model_init_kwargs: dict[str, Any]) -> PreTrainedModel:
    """Handles instantiation of model and tokenizer given a path and a set of configs."""
    auto_config = AutoConfig.from_pretrained(model_path)
    if any("ForCausalLM" in architecture for architecture in auto_config.architectures):
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_init_kwargs)
    elif auto_config.is_encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **model_init_kwargs)
    else:
        raise CycleModelError(
            f"Unsupported or unrecognised model type {auto_config.model_type} for model {model_path}. "
            "Make sure the provided model is either CausalLM or Seq2SeqLM. "
            "If you are using a custom model, you may need to pass the instantiated model to "
            "CycleTrainer."
        )

    # TODO: Handle quantisation
    return model


@dataclass
class DataCollatorForLanguageModelingAndEval:
    """Data collator which allows for custom labels to be used when evaluating on model generations."""

    tokenizer: PreTrainedTokenizerBase
    padding: bool = True
    max_length: int | None = None
    is_training: bool = True  # Add flag to handle train vs eval differently

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            {"input_ids": [f["input_ids"] for f in features]},
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # During training, handle labels as usual for causal LM
        if self.is_training:
            batch["labels"] = batch["input_ids"].clone()
        # During eval, preserve the original labels for generation comparison
        else:
            if "labels" in features[0]:
                # Pad labels to the same length
                labels = [f["labels"] for f in features]
                max_label_length = max(len(label) for label in labels)
                padded_labels = []

                for label in labels:
                    padding_length = max_label_length - len(label)
                    if padding_length > 0:
                        # Use -100 for padding to match training behavior
                        padded_label = torch.cat(
                            [torch.tensor(label), torch.full((padding_length,), -100, dtype=torch.long)]
                        )
                    else:
                        padded_label = torch.tensor(label)
                    padded_labels.append(padded_label)

                batch["labels"] = torch.stack(padded_labels)

            # You might also want to preserve original text
            if "target_text" in features[0]:
                batch["target_text"] = [f["target_text"] for f in features]

        return batch


class PreTrainingSummary:
    """Provides a detailed summary of the training configuration and resource usage.

    Currently displays:
    - Mode (Multi-Adapter or Dual-Model)
    - Model name and path
    - Memory usage
    - Parameter counts, trainable count and trainable percentage
    - Dataset sizes
    - Data types
    """

    def __init__(
        self,
        models: PreTrainedModel | dict[str, PreTrainedModel],
        model_configs: AutoConfig | dict[str, AutoConfig],
        datasets: dict[str, Dataset] | None = None,
        is_multi_adapter: bool = False,
    ):
        self.models = models if isinstance(models, dict) else {"model": models}
        self.model_configs = model_configs if isinstance(model_configs, dict) else {"model": model_configs}
        self.datasets = datasets or {}
        self.is_multi_adapter = is_multi_adapter

    def _get_memory_stats(self, model: PreTrainedModel) -> dict:
        """Get memory usage for a model after forcing garbage collection."""
        gc.collect()
        torch.cuda.empty_cache()

        if not torch.cuda.is_available():
            return {"allocated": 0, "cached": 0}

        return {
            "allocated": torch.cuda.memory_allocated() / (1024**2),  # MB
            "cached": torch.cuda.memory_reserved() / (1024**2),  # MB
        }

    def _count_parameters(self, model: PreTrainedModel) -> tuple[int, int]:
        """Count total and trainable parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return trainable_params, total_params

    def _get_dtype_info(self, model: PreTrainedModel) -> dict:
        """Get dtype information for model weights, gradients, and adapters."""
        dtypes = {"weights": next(model.parameters()).dtype}

        # Check gradient dtype if any parameter has gradients
        for param in model.parameters():
            if param.grad is not None:
                dtypes["gradients"] = param.grad.dtype
                break
        else:
            dtypes["gradients"] = None

        # Check for adapter dtypes if using PEFT
        if hasattr(model, "peft_config"):
            adapter_dtype = None
            for name, param in model.named_parameters():
                if "lora" in name.lower():  # Check for LoRA parameters
                    adapter_dtype = param.dtype
                    break
            dtypes["adapters"] = adapter_dtype

        return dtypes

    def _format_size(self, num: float) -> str:
        """Format large numbers with K, M, B suffixes."""
        for unit in ["", "K", "M", "B"]:
            if abs(num) < 1000:
                return f"{num:.1f}{unit}"
            num /= 1000
        return f"{num:.1f}T"

    def __str__(self) -> str:
        """Generate a formatted summary string."""
        CONTENT_WIDTH = 66  # Width between borders (total 68 with borders)
        BULLET = "-"  # Using simple hyphen for consistency

        def format_line(line: str) -> str:
            """Helper function to ensure proper line formatting with borders."""
            if line.startswith("╔") or line.startswith("╟") or line.startswith("╠") or line.startswith("╚"):
                return line

            # Strip existing right border if present
            content = line[:-1] if line.endswith("║") else line

            # If line starts with left border, remove it for length calculation
            if content.startswith("║"):
                inner_content = content[1:]
                # Calculate padding needed
                padding = CONTENT_WIDTH - len(inner_content)
                # Reconstruct line with proper padding
                return f"║{inner_content}{' ' * padding}║"
            else:
                # For lines without borders (shouldn't happen in this case)
                return f"║{content}{' ' * (CONTENT_WIDTH - len(content))}║"

        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════╗",
            "║                     TRAINING RUN SUMMARY                     ║",
            "╠══════════════════════════════════════════════════════════════╣",
        ]

        # Training mode
        mode = "Multi-Adapter Training" if self.is_multi_adapter else "Dual-Model Training"
        first_config = next(iter(self.model_configs.values()))
        model_info = f" Mode: {mode} ({first_config.name_or_path})"
        lines.append(format_line(f"║{model_info}"))
        lines.append("╟──────────────────────────────────────────────────────────────╢")

        # Model information
        for model_name, model in self.models.items():
            lines.append(format_line(f"║ Model: {model_name}"))

            # Memory usage
            mem_stats = self._get_memory_stats(model)
            lines.append(format_line(f"║   {BULLET} Memory Allocated: {mem_stats['allocated']:.1f}MB"))
            lines.append(format_line(f"║   {BULLET} Memory Cached: {mem_stats['cached']:.1f}MB"))

            # Parameter counts
            trainable, total = self._count_parameters(model)
            param_line = (
                f"║   {BULLET} Parameters: {self._format_size(total)} total || "
                f"{self._format_size(trainable)} trainable || "
                f"{trainable/total*100:.1f}%"
            )
            lines.append(format_line(param_line))

            # Dataset sizes
            dataset_key = f"{model_name}_train" if model_name in ["A", "B"] else "train"
            if dataset_key in self.datasets:
                try:
                    size = len(self.datasets[dataset_key])
                    lines.append(format_line(f"║   {BULLET} Training Samples: {size}"))
                except TypeError:
                    lines.append(format_line(f"║   {BULLET} Training Samples: N/A (dataset length not implemented)"))

            # Data types
            dtypes = self._get_dtype_info(model)
            lines.append(format_line(f"║   {BULLET} Weight dtype: {dtypes['weights']}"))
            lines.append(format_line(f"║   {BULLET} Gradient dtype: {dtypes['gradients'] or 'N/A'}"))
            if "adapters" in dtypes:
                lines.append(format_line(f"║   {BULLET} Adapter dtype: {dtypes['adapters'] or 'N/A'}"))

            lines.append("╟──────────────────────────────────────────────────────────────╢")

        lines[-1] = "╚══════════════════════════════════════════════════════════════╝"

        return "\n".join(lines)
