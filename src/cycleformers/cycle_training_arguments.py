from dataclasses import dataclass

from transformers.training_args import TrainingArguments


@dataclass
class CycleTrainingArguments(TrainingArguments):
    """Will eventually contain cycle-specific arguments"""
    pass


@dataclass
class ModelTrainingArguments:
    """Will eventually contain model-specific arguments"""
    pass


__all__ = ["CycleTrainingArguments", "ModelTrainingArguments"]
