from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from transformers.training_args import TrainingArguments


if TYPE_CHECKING:
    from .model_config import ModelConfig


@dataclass
class CycleTrainingArguments(TrainingArguments):
    """Will eventually contain cycle-specific arguments"""

    use_macct: bool = False
    report_to: list[str] = field(default_factory=lambda: ["wandb"])

    def __post_init__(self):
        super().__post_init__()
        self._model_config = None

    @property
    def model_config(self) -> "ModelConfig":
        return self._model_config

    @model_config.setter
    def model_config(self, value: "ModelConfig"):
        self._model_config = value


@dataclass
class ModelTrainingArguments:
    """Will eventually contain model-specific arguments"""

    pass


__all__ = ["CycleTrainingArguments", "ModelTrainingArguments"]
