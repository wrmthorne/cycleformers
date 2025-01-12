from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk

from cycleformers.cycle_trainer_utils import EvalGeneration


@dataclass
class ProcessorConfig:
    """Configuration class for dataset processors.

    This class defines the common configuration parameters used across different processors.
    Each specific processor can extend this with additional task-specific parameters.
    """

    dataset_name: str | PathLike[str] | None = None
    eval_split_ratio: float = 0.2
    dataset_seed: int = 42
    cache_dir: str | None = None
    split: list[str] | str | None = None
    max_samples: int | None = None
    streaming: bool = False  # FIXME: not a key feature right now - taking too much time
    evaluation_metrics: list[str] | None = None


class BaseProcessor(ABC):
    """Base class for dataset processors.

    This abstract base class defines the interface and common functionality for all dataset processors.
    Each specific format/task processor should inherit from this class and implement the required methods.

    The processor handles:
    1. Loading source datasets
    2. Converting to cycleformers format by splitting into two separate datasets A and B
    3. Applying common transformations (train/val splitting, shuffling)

    If the dataset already has a train/val/test split, those splits will be preserved and only train will be made
    non-parallel.

    Args:
        config: Configuration object controlling processor behavior. If not provided, uses default CONFIG_CLS.

    Example:
        >>> config = ProcessorConfig(eval_split_ratio=0.2, dataset_seed=42)
        >>> processor = ConcreteProcessor(config)
        >>> dataset_A, dataset_B = processor.process()
        >>> print(dataset_A.keys())
        dict_keys(['train', 'test'])
    """

    def __init__(self, config: ProcessorConfig = ProcessorConfig()):
        self.config = config

    def load(self) -> DatasetDict:
        """Load the source dataset. Override for custom loading logic."""
        if self.config.dataset_name is None:
            raise ValueError("No dataset name was provided. Cannot load `None`.")

        if Path(self.config.dataset_name).exists():
            return load_from_disk(self.config.dataset_name)

        return load_dataset(
            self.config.dataset_name,
            cache_dir=self.config.cache_dir,
            streaming=self.config.streaming,
        )

    @abstractmethod
    def preprocess(self, dataset: DatasetDict) -> DatasetDict:
        """Preprocess the dataset into two separate datasets A and B."""
        raise NotImplementedError

    @abstractmethod
    def compute_metrics(self) -> dict[str, Callable[[EvalGeneration], dict[str, float]]]:
        """Return single callable or dict of callables corresponding to the task in a dataset."""
        raise NotImplementedError

    def process(self) -> DatasetDict:
        """Process the dataset into two separate datasets A and B.

        Returns:
            Tuple[DatasetDict, DatasetDict]: Two datasets A and B, each containing 'train' and 'test' splits
        """
        dataset = self.load()

        if not isinstance(dataset, DatasetDict):
            dataset = DatasetDict({"train": dataset})

        if dataset.keys() == ["train"]:
            dataset = dataset.train_test_split(test_size=self.config.eval_split_ratio, seed=self.config.dataset_seed)

        dataset_A, dataset_B = self.preprocess(dataset)

        dataset_A["train"] = dataset_A["train"].shuffle(seed=self.config.dataset_seed)
        dataset_B["train"] = dataset_B["train"].shuffle(seed=self.config.dataset_seed + 1)

        return dataset_A, dataset_B
