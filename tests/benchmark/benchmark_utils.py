from abc import ABC, abstractmethod
from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

from cycleformers import CycleTrainer, CycleTrainingArguments


@dataclass
class BenchmarkConfig:
    config_name: str
    output_dir: str = "benchmark_results/"


class Benchmark(ABC):
    def __init__(self, config: BenchmarkConfig, *args, **kwargs):
        self.config = config

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError


class MockCycleTrainer(CycleTrainer):
    """Lightweight mock of CycleTrainer for benchmarking"""

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.args = CycleTrainingArguments(output_dir="./tmp")
        self.tokenizer_A = self.tokenizer_B = tokenizer
        self.sep_seq = "\n\n"
