import json
import os
import timeit
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from random import Random

import torch
from benchmark_utils import Benchmark, BenchmarkConfig, MockCycleTrainer
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from cycleformers.cycles import _default_prepare_cycle_inputs, _prepare_causal_skip_cycle_inputs
from cycleformers.utils import DEFAULT_SEP_SEQ


os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class TokenPrepBenchmarkConfig(BenchmarkConfig):
    config_name: str = "token_prep"
    dataset_name: str
    model_A_name: str
    model_B_name: str
    cycle_name: str | None = None
    num_samples: int = 500
    batch_sizes: list[int] = field(default_factory=lambda: [1, 8, 32, 128])
    devices: list[str] = field(default_factory=lambda: ["cpu", "cuda"])
    seed: int = field(default_factory=lambda: int(datetime.now().timestamp()))


class TokenPreparationBenchmark:
    def __init__(self, config: TokenPrepBenchmarkConfig):
        self.config = config
        self.random = Random(config.seed)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_A_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model_A = AutoModelForCausalLM.from_pretrained(config.model_A_name)

        self.save_dir = Path(__file__).parent / "benchmark_results" / "token_prep"
        self.prepare_dataset()

    def prepare_dataset(self):
        # FIXME: Currently only supports datasets with "instruction", "response" as columns
        self.dataset = load_dataset(self.config.dataset_name, split="train").select(range(self.config.num_samples))
        self.dataset = self.dataset.map(
            lambda x: {
                "synth_ids": self.tokenizer(
                    x["instruction"] + DEFAULT_SEP_SEQ + x["response"], padding=False
                ).input_ids,
                "real_ids": self.tokenizer(x["instruction"], padding=False).input_ids,
            }
        )

    def manual_right_pad_batch(self, batch_ids: list[list[int]], pad_token_id: int) -> torch.Tensor:
        """Right pad a batch of token IDs to the maximum length in the batch."""
        max_length = max(len(ids) for ids in batch_ids)
        padded_batch = []

        for ids in batch_ids:
            padding_length = max_length - len(ids)
            padded_ids = ids + [pad_token_id] * padding_length
            padded_batch.append(padded_ids)

        return torch.tensor(padded_batch)

    def get_dataloader(self, batch_size: int, device: str):
        generator = torch.Generator(device=device)
        generator.manual_seed(self.config.seed)

        def collate_fn(batch):
            real_ids = [item["real_ids"] for item in batch]
            synth_ids = [item["synth_ids"] for item in batch]

            # Manually pad both sequences
            real_ids_padded = self.manual_right_pad_batch(real_ids, self.tokenizer.pad_token_id)
            synth_ids_padded = self.manual_right_pad_batch(synth_ids, self.tokenizer.pad_token_id)

            return real_ids_padded, synth_ids_padded

        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            # generator=generator,
            collate_fn=collate_fn,
        )
        return dataloader

    def _save_metrics(self, metrics: dict):
        run_dir = self.save_dir / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / f"metrics.json", "w") as f:
            json.dump(metrics, f)

    def run(self, implementations: list[str] | None = None, n_runs: int = 5):
        """Run benchmarks across all configurations"""

        results = []

        for batch_size in self.config.batch_sizes:
            for device in self.config.devices:
                if device == "cuda" and not torch.cuda.is_available():
                    continue

                print(f"\nBenchmarking batch_size={batch_size} on {device}")

                # Prepare data
                dataloader = self.get_dataloader(batch_size, device)

                for impl in implementations:
                    # Time the implementation

                    def run_impl():
                        outputs = []
                        for batch in dataloader:
                            real_ids, synth_ids = batch

                            output = impl(
                                real_input_ids=real_ids.to(device),
                                synth_input_ids=synth_ids.to(device),
                                model_gen=self.model_A,
                                model_train=self.model_A,
                                tokenizer_gen=self.tokenizer,
                                tokenizer_train=self.tokenizer,
                                cycle_name="A",
                            )
                            outputs.append(output)
                        return outputs

                    # Time runs
                    if device == "cuda":
                        torch.cuda.synchronize()

                    timer = timeit.Timer(run_impl)
                    times = timer.repeat(repeat=n_runs, number=5)

                    metrics = {"mean": sum(times) / len(times), "min": min(times), "max": max(times)}

                    results.append({"implementation": impl, "batch_size": batch_size, "device": device, **metrics})

                    print(f"\n{impl}:")
                    print(f"Mean: {metrics['mean']*1000:.2f}ms")
                    print(f"Min: {metrics['min']*1000:.2f}ms")
                    print(f"Max: {metrics['max']*1000:.2f}ms")

        return results


if __name__ == "__main__":
    config = TokenPrepBenchmarkConfig(
        dataset_name="MBZUAI/LaMini-instruction",
        model_A_name="trl-internal-testing/tiny-LlamaForCausalLM-3.1",
        model_B_name="trl-internal-testing/tiny-LlamaForCausalLM-3.1",
    )

    benchmark = TokenPreparationBenchmark(config=config)
    benchmark.run(implementations=[_default_prepare_cycle_inputs, _prepare_causal_skip_cycle_inputs])
