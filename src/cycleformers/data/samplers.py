import datasets
import torch
from torch.utils.data import IterableDataset, Sampler

from .datasets import DatasetConfig, PeftDatasetConfig
from .utils import StopOn


# TODO: Will likely need to be reviewed for performance and compatibility in the future
class CyclicSampler(Sampler):
    """Samples from multiple datasets in a cyclic fashion"""

    def __init__(
        self,
        *dataset_configs: list[DatasetConfig | PeftDatasetConfig],
        batch_size: int,
        stop_on: StopOn = StopOn.FIRST,
    ):
        self.datasets = dataset_configs
        self.batch_size = batch_size
        self.stop_on = stop_on

    def __iter__(self):
        # Create iterators for each dataset
        active_iterators = {}
        for dataset in self.datasets:
            name = dataset.dataset_name
            if isinstance(dataset.dataset, (IterableDataset, datasets.Dataset)):
                active_iterators[name] = (iter(dataset.dataset), dataset.batch_extras)
            else:
                active_iterators[name] = (iter(torch.randperm(len(dataset.dataset)).tolist()), dataset.batch_extras)

        # Cycle through datasets until one is depleted
        # TODO: Test if this throws an error if the last batch is not complete
        while active_iterators:
            for name, (iterator, batch_extras) in list(active_iterators.items()):
                try:
                    batch = []
                    for _ in range(self.batch_size):
                        idx = next(iterator)
                        batch.append({**batch_extras, "idx": idx})
                    yield batch
                except StopIteration:
                    if self.stop_on == StopOn.LAST:
                        active_iterators.pop(name)
                    else:
                        return

    def __len__(self):
        return len(self.datasets) * (min(len(dataset.dataset) for dataset in self.datasets) // self.batch_size)
