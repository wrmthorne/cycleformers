import cProfile
import pstats
import time
from pathlib import Path

import torch.profiler
from memory_profiler import profile as memory_profile
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, HfArgumentParser

from cycleformers import CycleTrainer, CycleTrainingArguments, ModelConfig


class ProfilingCycleTrainer(CycleTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profiling_stats = {}

    def train(self):
        # Profile entire training run
        profiler = cProfile.Profile()
        profiler.enable()

        # Setup PyTorch profiler for detailed GPU analysis
        pytorch_profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./cycle_training_profiles"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

        pytorch_profiler.start()

        try:
            result = super().train()

            # Collect profiling data
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.sort_stats("cumtime")

            # Create unique output directory
            output_dir = Path("cycle_training_profiles")
            output_dir.mkdir(exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            profile_path = output_dir / f"cycle_training_profile_{timestamp}.prof"

            stats.dump_stats(str(profile_path))

            return result
        finally:
            pytorch_profiler.stop()

    def _cycle_step(self, *args, **kwargs):
        step_start_time = time.perf_counter()

        with torch.profiler.record_function("cycle_step_total"):
            # Call the original implementation
            metrics = super()._cycle_step(*args, **kwargs)

        # Track step timing
        step_duration = time.perf_counter() - step_start_time
        if "step_times" not in self.profiling_stats:
            self.profiling_stats["step_times"] = []
        self.profiling_stats["step_times"].append(step_duration)

        return metrics

    def analyze_performance(self):
        """Analyze collected performance metrics"""
        if self.profiling_stats.get("step_times"):
            avg_step_time = sum(self.profiling_stats["step_times"]) / len(self.profiling_stats["step_times"])
            print(f"Average step time: {avg_step_time:.4f} seconds")

            # Load and analyze the cProfile data
            stats = pstats.Stats("cycle_training_profile.prof")
            print("\nTop 10 time-consuming functions:")
            stats.sort_stats("cumtime").print_stats(10)

            # Memory analysis tips
            print("\nMemory usage peaks can be found in the memory_profiler output above")
            print("Check PyTorch profiler traces in TensorBoard for detailed GPU memory analysis")


from cycleformers import CycleTrainer, CycleTrainingArguments, ModelConfig
from cycleformers.import_utils import is_liger_kernel_available
from cycleformers.task_processors.ner import CONLL2003Processor, CONLL2003ProcessorConfig
from cycleformers.utils import get_peft_config


def get_model_and_tokenizer(model_config, training_args):
    """Initialize model and tokenizer from config"""
    auto_config = AutoConfig.from_pretrained(model_config.model_name_or_path)
    if not auto_config.is_encoder_decoder:
        model_class = AutoModelForCausalLM
    else:
        model_class = AutoModelForSeq2SeqLM

    model = model_class.from_pretrained(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        trust_remote_code=model_config.trust_remote_code,
        # use_liger_kernel=training_args.use_liger_kernel and is_liger_available(),
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    return model, tokenizer


def main():
    # FIXME: Work out how to get dataclass_A.param_A into form dataclass.param
    parser = HfArgumentParser((CycleTrainingArguments, ModelConfig, CONLL2003ProcessorConfig))

    maybe_yaml = Path("tests/benchmark/profiler_configs/causal.yaml")
    if maybe_yaml.suffix == ".yaml" and maybe_yaml.exists():
        training_args, model_config, conll_config = parser.parse_yaml_file(maybe_yaml)
    else:
        raise ValueError("Only support for yaml right now")

    task_processor = CONLL2003Processor(conll_config)
    dataset_A, dataset_B = task_processor.process()

    models, tokenizer = get_model_and_tokenizer(model_config, training_args)
    if not model_config.use_peft:
        model_B, _ = get_model_and_tokenizer(model_config, training_args)
        models = {"A": models, "B": model_B}

    trainer = ProfilingCycleTrainer(
        args=training_args,
        models=models,
        tokenizers=tokenizer,
        train_dataset_A=dataset_A["train"],
        train_dataset_B=dataset_B["train"],
        eval_dataset_A=dataset_A["eval"] if not training_args.eval_strategy == "no" else None,
        eval_dataset_B=dataset_B["eval"] if not training_args.eval_strategy == "no" else None,
        peft_configs=get_peft_config(model_config),
    )
    trainer.train()

    trainer.analyze_performance()


if __name__ == "__main__":
    main()
