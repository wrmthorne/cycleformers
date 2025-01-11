import gc
from dataclasses import dataclass
from os import PathLike
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, PreTrainedModel

from cycleformers.exceptions import CycleModelError


@dataclass
class EvalGeneration:
    """Class for storing evaluation generation results."""

    predictions: list[str]
    labels: list[str]


def load_model(model_path: str | PathLike[str], **model_init_kwargs: dict[str, Any]) -> PreTrainedModel:
    auto_config = AutoConfig.from_pretrained(model_path)
    if "ForCausalLM" in auto_config.model_type:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_init_kwargs)
    elif auto_config.is_encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **model_init_kwargs)
    else:
        raise CycleModelError(
            "Unsupported or unrecognised model type. Make sure the provided model is either "
            "CausalLM or Seq2SeqLM. If you are using a custom model, you may need to pass the instantiated model to "
            "CycleTrainer."
        )

    # TODO: Handle quantisation
    return model


class PreTrainingSummary:
    """Provides a detailed summary of the training configuration and resource usage."""

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
        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════╗",
            "║                     TRAINING RUN SUMMARY                     ║",
            "╠══════════════════════════════════════════════════════════════╣",
        ]

        # Training mode
        mode = "Multi-Adapter Training" if self.is_multi_adapter else "Standard Training"
        # Get the first config's name or path
        first_config = next(iter(self.model_configs.values()))
        lines.append(f"║ Mode: {mode:<54} ({first_config.name_or_path}) ║")
        lines.append("╟──────────────────────────────────────────────────────────────╢")

        # Model information
        for model_name, model in self.models.items():
            lines.append(f"║ Model: {model_name:<53} ({model}) ║")

            # Memory usage
            mem_stats = self._get_memory_stats(model)
            lines.append(f"║   • Memory Allocated: {mem_stats['allocated']:.1f}MB")
            lines.append(f"║   • Memory Cached: {mem_stats['cached']:.1f}MB")

            # Parameter counts
            trainable, total = self._count_parameters(model)
            lines.append(
                f"║   • Parameters: {self._format_size(total)} total || "
                f"{self._format_size(trainable)} trainable || "
                f"{trainable/total*100:.1f}%"
            )

            # Dataset sizes
            dataset_key = f"{model_name}_train" if model_name in ["A", "B"] else "train"
            if dataset_key in self.datasets:
                try:
                    size = len(self.datasets[dataset_key])
                    lines.append(f"║   • Training Samples: {size}")
                except TypeError:
                    lines.append("║   • Training Samples: N/A (dataset length not implemented)")

            # Data types
            dtypes = self._get_dtype_info(model)
            lines.append(f"║   • Weight dtype: {dtypes['weights']}")
            lines.append(f"║   • Gradient dtype: {dtypes['gradients'] or 'N/A'}")
            if "adapters" in dtypes:
                lines.append(f"║   • Adapter dtype: {dtypes['adapters'] or 'N/A'}")

            lines.append("╟──────────────────────────────────────────────────────────────╢")

        lines[-1] = "╚══════════════════════════════════════════════════════════════╝"

        summary = "\n".join(line + " " * (69 - len(line)) + "║" if "║" not in line[-2:] else line for line in lines)
        print(summary)
        return summary
