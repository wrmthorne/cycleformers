from dataclasses import dataclass, field

import datasets
from torch.utils.data import Dataset, IterableDataset


@dataclass
class MACCTConfig:
    """Configuration for Multi-Adapter Cycle-Consistency Training (MACCT)"""

    datasets: dict[str, Dataset | IterableDataset | "datasets.Dataset"] = field(default_factory=dict)
    per_dataset_batch_size: dict[str, int] | None = None
