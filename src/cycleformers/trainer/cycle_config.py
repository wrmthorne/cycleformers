from dataclasses import dataclass, field

from ..data.utils import StopOn


@dataclass
class CycleConfig:
    stop_on: StopOn = field(default=StopOn.FIRST, metadata={"help": "Dataset condition for training terminating"})
