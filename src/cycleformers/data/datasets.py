import warnings
from dataclasses import dataclass, field

import datasets
from torch.utils.data import Dataset

from ..utils import encode_name_for_forward


# TODO: Add support for per_device_batch_size
@dataclass
class DatasetConfig:
    dataset: datasets.Dataset | Dataset | None = field(
        default=None,
        metadata={
            "help": "The dataset to use for training or evaluation. Can be a HuggingFace dataset or PyTorch dataset."
        },
    )
    dataset_name: str | None = field(
        default=None,
        metadata={
            "help": "A unique name to identify this dataset. Used for logging and to match datasets with adapters."
        },
    )

    def __post_init__(self):
        if self.dataset is None:
            raise ValueError("dataset must be provided")

        if self.dataset_name is None:
            self.dataset_name = str(len(self.dataset_indices))

        self.dataset_idx = encode_name_for_forward(self.dataset_name)

    @property
    def batch_extras(self):
        return {"dataset_idx": self.dataset_idx}


@dataclass
class PeftDatasetConfig(DatasetConfig):
    train_adapter: str = field(
        default_factory=lambda: "",
        metadata={
            "help": "The name of the adapter that will be trained when this dataset is used as real input data during cycle training."
        },
    )

    def __post_init__(self):
        super().__post_init__()

        if self.train_adapter == "":
            raise ValueError("train_adapter must be provided")

        if hasattr(self.dataset, "train_adapter"):
            warnings.warn(
                f"train_adapter is specified in dataset. This instance will be ignored and {self.train_adapter} will be used instead."
            )

        if self.dataset_name == self.train_adapter:
            warnings.warn(
                f"train_adapter should refer to the adapter being trained but dataset_name is also set to {self.train_adapter}. Adapter {self.train_adapter} may erroneously receive inputs from this dataset. Check that you are passing the correct dataset_name and adapter_name. See https://github.com/wrmthorne/CycleFormers/blob/main/README.md for examples of best practice."
            )

        self.train_adapter_idx = encode_name_for_forward(self.train_adapter)

    @property
    def batch_extras(self):
        return {"target_adapter_idx": self.train_adapter_idx}


__all__ = ["DatasetConfig", "PeftDatasetConfig"]
