from dataclasses import dataclass

from transformers import TrainingArguments


@dataclass
class CycleTrainingArguments(TrainingArguments):
    model_A_args: TrainingArguments
    model_B_args: TrainingArguments

    # Will be expanded as necessary