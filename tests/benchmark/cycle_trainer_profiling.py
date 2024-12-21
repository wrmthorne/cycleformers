import cProfile
import logging
import pstats
import sys
import time
from pathlib import Path

import torch.cuda as cuda
import torch.profiler
import yaml
from memory_profiler import profile as memory_profile
from profiler_utils import record_function_wrapper
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from cycleformers import CfArgumentParser, CycleTrainer, CycleTrainingArguments, ModelConfig
from cycleformers.import_utils import is_liger_kernel_available
from cycleformers.model_config import ModelConfigA, ModelConfigB, merge_configs
from cycleformers.task_processors.ner import CONLL2003Processor, CONLL2003ProcessorConfig
from cycleformers.utils import VALID_LIGER_MODELS, get_peft_config, print_trainable_params


logger = logging.getLogger(__file__)

MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 1_000_000


class ProfilingCycleTrainer(CycleTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profiling_stats = {
            "step_times": [],
            "gpu_memory": [],
            "max_gpu_memory": 0,
        }

        # Create unique output directory for this run
        self.profile_output_dir = (
            Path(__file__).parent / "profiles" / f'cycle_trainer--{time.strftime("%Y%m%d_%H%M%S")}'
        )
        self.profile_output_dir.mkdir(parents=True, exist_ok=True)

        self.profile_path = self.profile_output_dir / "profile.prof"
        self.memory_path = self.profile_output_dir / "memory_profile.txt"
        self.cuda_memory_path = self.profile_output_dir / "cuda_memory_snapshots"

        # Save args to yaml file
        with open(self.profile_output_dir / "profiler_args.yaml", "w") as f:
            yaml.dump(self.args, f)

        print("=" * 40)
        print("Model A: ", end="")
        print_trainable_params(self.model_A)
        if not self.is_macct_model:
            print("Model B: ", end="")
            print_trainable_params(self.model_B)
        print("=" * 40)

    def _log_gpu_memory(self):
        """Log current GPU memory usage"""
        if not cuda.is_available():
            return None

        current_memory = cuda.memory_allocated() / 1024**2  # Convert to MB
        max_memory = cuda.max_memory_allocated() / 1024**2  # Convert to MB

        self.profiling_stats["gpu_memory"].append(current_memory)
        self.profiling_stats["max_gpu_memory"] = max(self.profiling_stats["max_gpu_memory"], max_memory)

        return current_memory, max_memory

    def train(self):
        torch.cuda.memory._record_memory_history(max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT)

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
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.profile_output_dir / "tensorboard_trace"),
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

            # Save profiling data
            stats.dump_stats(str(self.profile_path))

            # Save memory statistics
            with open(self.memory_path, "w") as f:
                f.write(f"Max GPU Memory Used: {self.profiling_stats['max_gpu_memory']:.2f} MB\n")
                f.write("GPU Memory Usage per Step:\n")
                for i, mem in enumerate(self.profiling_stats["gpu_memory"]):
                    f.write(f"Step {i}: {mem:.2f} MB\n")

            return result
        finally:
            pytorch_profiler.stop()
            torch.cuda.memory._record_memory_history(enabled=None)
            try:
                torch.cuda.memory._dump_snapshot(f"{self.cuda_memory_path}.pickle")
            except Exception as e:
                logger.error(f"Failed to capture memory snapshot {e}")

    @record_function_wrapper("## Cycle Step ##")
    def _cycle_step(self, *args, **kwargs):
        return super()._cycle_step(*args, **kwargs)

    @record_function_wrapper("## Prepare Cycle Inputs ##")
    def _prepare_cycle_inputs(self, *args, **kwargs):
        return super().prepare_cycle_inputs(*args, **kwargs)

    def analyze_performance(self):
        """Analyze collected performance metrics"""
        if self.profiling_stats.get("step_times"):
            avg_step_time = sum(self.profiling_stats["step_times"]) / len(self.profiling_stats["step_times"])
            print(f"Average step time: {avg_step_time:.4f} seconds")

            # GPU Memory Statistics
            if cuda.is_available():
                print("\nGPU Memory Statistics:")
                print(f"Peak GPU memory usage: {self.profiling_stats['max_gpu_memory']:.2f} MB")
                avg_memory = sum(self.profiling_stats["gpu_memory"]) / len(self.profiling_stats["gpu_memory"])
                print(f"Average GPU memory usage: {avg_memory:.2f} MB")
                print(f"Current GPU memory usage: {cuda.memory_allocated() / 1024**2:.2f} MB")
                print(f"Current GPU memory cached: {cuda.memory_reserved() / 1024**2:.2f} MB")

            # Load and analyze the cProfile data
            stats = pstats.Stats(str(self.profile_output_dir / "profile.prof"))
            print("\nTop 10 time-consuming functions:")
            stats.sort_stats("cumtime").print_stats(10)

            # Memory analysis tips
            print(f"\nProfiling data has been saved to: {self.profile_output_dir}")
            print("Check PyTorch profiler traces in TensorBoard for detailed GPU memory analysis")


if is_liger_kernel_available():
    from liger_kernel.transformers import AutoLigerKernelForCausalLM


def get_model_and_tokenizer(model_config, training_args):
    """Initialize model and tokenizer from config"""
    config = AutoConfig.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
    )
    config.use_cache = False

    model_kwargs = {}

    if not config.is_encoder_decoder:
        if is_liger_kernel_available() and model_config.use_liger and config.model_type in VALID_LIGER_MODELS:
            model_class = AutoLigerKernelForCausalLM
            model_kwargs["use_liger_kernel"] = training_args.use_liger_kernel
        else:
            model_class = AutoModelForCausalLM
    else:
        model_class = AutoModelForSeq2SeqLM

    model = model_class.from_pretrained(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
        config=config,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        device_map="auto",
    )

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # Print the actual dtype of the first parameter
    print(f"Model weights dtype: {next(model.parameters()).dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    return model, tokenizer


def main():
    sys.argv = [__file__, str(Path(__file__).parent / "profiler_configs/causal.yaml")]
    parser = CfArgumentParser(
        (CycleTrainingArguments, ModelConfig, ModelConfigA, ModelConfigB, CONLL2003ProcessorConfig), task="train"
    )
    args, model_config_base, model_config_A, model_config_B, conll_config = parser.parse_args_and_config()
    model_config_base = merge_configs(model_config_base, model_config_A, model_config_B)
    args.model_config = model_config_base

    task_processor = CONLL2003Processor(conll_config)
    dataset_A, dataset_B = task_processor.process()

    model_A, tokenizer_A = get_model_and_tokenizer(args.model_config.A, args)

    # Train by adapter swapping
    if not args.use_macct:
        # Get model B using merged B config
        model_B, tokenizer_B = get_model_and_tokenizer(args.model_config.B, args)
        models = {"A": model_A, "B": model_B}
        tokenizers = {"A": tokenizer_A, "B": tokenizer_B} if tokenizer_A != tokenizer_B else tokenizer_A
    else:
        models = model_A
        tokenizers = tokenizer_A

    trainer = ProfilingCycleTrainer(
        args=args,
        models=models,
        tokenizers=tokenizers,
        train_dataset_A=dataset_A["train"],
        train_dataset_B=dataset_B["train"],
        eval_dataset_A=dataset_A["eval"] if not args.eval_strategy == "no" else None,
        eval_dataset_B=dataset_B["eval"] if not args.eval_strategy == "no" else None,
        peft_configs=get_peft_config(model_config_base),
    )

    trainer.train()

    trainer.analyze_performance()


if __name__ == "__main__":
    main()
