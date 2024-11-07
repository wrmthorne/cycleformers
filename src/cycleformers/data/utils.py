from enum import Enum


class StopOn(Enum):
    """Controls when sampling stops when datasets have different lengths.
    FIRST: Stop sampling when the first dataset runs out of samples
    LAST: Continue sampling until all datasets run out of samples
    """

    FIRST = "FIRST"
    LAST = "LAST"
