from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class DataConfig:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str | None = None
    dataset_config_name: str | None = None
    text_column: str = "text"
    formatting_func: Callable | None = None
    max_seq_length: int | None = None
    remove_unused_columns: bool = True
    preprocessing_num_workers: int | None = None
