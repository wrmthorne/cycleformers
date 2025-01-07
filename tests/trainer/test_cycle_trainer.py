import json
import shutil
from pathlib import Path

import pytest
import torch
from datasets import Dataset

from cycleformers import CycleTrainer, CycleTrainingArguments
from cycleformers.cycles import _default_prepare_cycle_inputs, _prepare_causal_skip_cycle_inputs
from cycleformers.import_utils import is_peft_available


if is_peft_available():
    from peft import PeftModel
    from peft.tuners.lora import LoraConfig


class TrainerTestMixin:
    """Base mixin for trainer tests providing common fixtures and utilities."""

    @pytest.fixture(autouse=True)
    def setup(self, temp_dir):
        """Setup common test environment with temporary output directory."""
        self.output_dir = temp_dir / "trainer_test"
        self.output_dir.mkdir(parents=True)

        # Common training arguments used across tests
        self.default_args = CycleTrainingArguments(
            output_dir=str(self.output_dir),
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=1,
            save_strategy="steps",
            save_steps=1,
        )

        yield

        # Cleanup
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    @pytest.fixture
    def text_dataset(self):
        """Create a simple text dataset for testing."""
        return Dataset.from_dict({"text": ["Hello world", "How are you?", "Testing is fun"]})

    @pytest.fixture
    def default_trainer(self, random_model_and_tokenizer, text_dataset):
        """Create a default trainer instance with common configuration."""
        model, tokenizer = random_model_and_tokenizer
        return CycleTrainer(
            args=self.default_args,
            models={"A": model, "B": model.clone()},
            tokenizers=tokenizer,
            train_dataset_A=text_dataset,
            train_dataset_B=text_dataset,
        )

    def assert_checkpoint_exists(self, checkpoint_dir: Path, is_macct: bool = False):
        """Assert that a checkpoint directory contains all expected files."""
        if is_macct:
            # MACCT model saves adapters and shared tokenizer
            assert (checkpoint_dir / "A" / "adapter_model.safetensors").exists()
            assert (checkpoint_dir / "B" / "adapter_model.safetensors").exists()
            assert (checkpoint_dir / "A" / "adapter_config.json").exists()
            assert (checkpoint_dir / "B" / "adapter_config.json").exists()
            assert (checkpoint_dir / "tokenizer_config.json").exists()
        else:
            # Regular model saves full models and tokenizers
            assert (checkpoint_dir / "A" / "model.safetensors").exists()
            assert (checkpoint_dir / "B" / "model.safetensors").exists()
            assert (checkpoint_dir / "tokenizer_config.json").exists()

        # Common files for both types
        assert (checkpoint_dir / "A" / "optimizer.pt").exists()
        assert (checkpoint_dir / "B" / "optimizer.pt").exists()
        assert (checkpoint_dir / "A" / "scheduler.pt").exists()
        assert (checkpoint_dir / "B" / "scheduler.pt").exists()
        assert (checkpoint_dir / "rng_state.pth").exists()
        assert (checkpoint_dir / "trainer_state.json").exists()


# TODO: Rename when better defined
@pytest.fixture(name="gen_peft_model")
def fixture_gen_peft_model(request):
    return request.param


@pytest.mark.skipif(not is_peft_available(), reason="PEFT is required for MACCT tests")
class TestCycleTrainerMACCT(TrainerTestMixin):
    """Tests for MACCT (Multi-Adapter Cycle Consistency Training) functionality."""

    @pytest.mark.parametrize(
        "peft_config",
        [
            None,  # Default LORA config
            LoraConfig(r=8, lora_alpha=32),  # Custom single config
            {"A": LoraConfig(r=8, lora_alpha=16), "B": LoraConfig(r=16, lora_alpha=32)},  # Different configs
        ],
    )
    def test_use_macct_with_single_model(self, causal_model, causal_tokenizer, peft_config, text_dataset):
        """Test MACCT initialization with different PEFT configurations."""
        args = self.default_args
        args.use_macct = True

        trainer = CycleTrainer(
            args=args,
            models=causal_model,
            tokenizers=causal_tokenizer,
            train_dataset_A=text_dataset,
            train_dataset_B=text_dataset,
            peft_configs=peft_config,
        )

        # Verify MACCT setup
        assert trainer.is_macct_model
        assert trainer.model_A is trainer.model_B
        assert isinstance(trainer.model_A, PeftModel)
        assert isinstance(trainer.model_B, PeftModel)
        assert trainer.model_A.get_adapter("A") is not None
        assert trainer.model_B.get_adapter("B") is not None

        # Verify adapter configurations
        if peft_config is None:
            assert trainer.model_A.get_adapter("A").r == 8
            assert trainer.model_B.get_adapter("B").r == 8
        elif isinstance(peft_config, LoraConfig):
            assert trainer.model_A.get_adapter("A").r == peft_config.r
            assert trainer.model_B.get_adapter("B").r == peft_config.r
        elif isinstance(peft_config, dict):
            assert trainer.model_A.get_adapter("A").r == peft_config["A"].r
            assert trainer.model_B.get_adapter("B").r == peft_config["B"].r

    @pytest.mark.slow
    def test_macct_training_cycle(self, any_model_and_tokenizer, text_dataset):
        """Test a complete MACCT training cycle."""
        args = self.default_args
        args.use_macct = True
        args.num_train_epochs = 1
        args.max_steps = 2

        model, tokenizer = any_model_and_tokenizer
        trainer = CycleTrainer(
            args=args,
            models=model,
            tokenizers=tokenizer,
            train_dataset_A=text_dataset,
            train_dataset_B=text_dataset,
        )
        trainer.train()

        checkpoint_dir = self.output_dir / "checkpoint-1"
        self.assert_checkpoint_exists(checkpoint_dir, is_macct=True)

    @pytest.mark.parametrize("save_only_model", [True, False], ids=["model_only", "full_checkpoint"])
    def test_macct_checkpoint_saving(self, causal_model, causal_tokenizer, text_dataset, save_only_model):
        """Test checkpoint saving behavior with MACCT models."""
        args = self.default_args
        args.use_macct = True
        args.save_only_model = save_only_model

        trainer = CycleTrainer(
            args=args,
            models=causal_model,
            tokenizers=causal_tokenizer,
            train_dataset_A=text_dataset,
            train_dataset_B=text_dataset,
        )

        # Save checkpoint
        trainer._save_checkpoint(trainer.model_A)
        checkpoint_dir = self.output_dir / "checkpoint-0"

        # Verify adapter files
        assert (checkpoint_dir / "A" / "adapter_model.safetensors").exists()
        assert (checkpoint_dir / "B" / "adapter_model.safetensors").exists()
        assert (checkpoint_dir / "A" / "adapter_config.json").exists()
        assert (checkpoint_dir / "B" / "adapter_config.json").exists()

        # Verify optimizer/scheduler files based on save_only_model
        if not save_only_model:
            assert (checkpoint_dir / "A" / "optimizer.pt").exists()
            assert (checkpoint_dir / "A" / "scheduler.pt").exists()
            assert (checkpoint_dir / "B" / "optimizer.pt").exists()
            assert (checkpoint_dir / "B" / "scheduler.pt").exists()
        else:
            assert not (checkpoint_dir / "A" / "optimizer.pt").exists()
            assert not (checkpoint_dir / "A" / "scheduler.pt").exists()
            assert not (checkpoint_dir / "B" / "optimizer.pt").exists()
            assert not (checkpoint_dir / "B" / "scheduler.pt").exists()


class TestSetCycleInputsFn(TrainerTestMixin):
    """Tests for cycle input preparation functionality."""

    def test_set_custom_fn(self, default_trainer):
        """Test setting a custom cycle inputs preparation function."""

        def test_fn(*args, **kwargs):
            return args, kwargs

        default_trainer.set_cycle_inputs_fn(test_fn)
        assert default_trainer._prepare_cycle_inputs.__func__ == test_fn

    def test_set_default_causal_skip(self, default_trainer):
        """Test default cycle inputs function for causal models."""
        default_trainer.set_cycle_inputs_fn()
        assert default_trainer._prepare_cycle_inputs.__func__ == _prepare_causal_skip_cycle_inputs

    def test_set_default_seq2seq(self, seq2seq_model_and_tokenizer, causal_model_and_tokenizer, text_dataset):
        """Test default cycle inputs function for mixed model types."""
        causal_model, causal_tokenizer = causal_model_and_tokenizer
        seq2seq_model, seq2seq_tokenizer = seq2seq_model_and_tokenizer

        trainer = CycleTrainer(
            args=self.default_args,
            models={"A": seq2seq_model, "B": causal_model},
            tokenizers={"A": seq2seq_tokenizer, "B": causal_tokenizer},
            train_dataset_A=text_dataset,
            train_dataset_B=text_dataset,
        )
        trainer.set_cycle_inputs_fn()
        assert trainer._prepare_cycle_inputs.__func__ == _default_prepare_cycle_inputs


class TestSaveCheckpoint(TrainerTestMixin):
    """Tests for checkpoint saving functionality."""

    def test_model_save(self, default_trainer):
        """Test saving model weights."""
        default_trainer._save_checkpoint(None, None)
        checkpoint_dir = self.output_dir / "checkpoint-0"

        assert (checkpoint_dir / "A" / "model.safetensors").exists()
        assert (checkpoint_dir / "B" / "model.safetensors").exists()

    def test_optimizer_scheduler_save(self, default_trainer):
        """Test saving optimizer and scheduler states."""
        default_trainer._save_checkpoint(None, None)
        checkpoint_dir = self.output_dir / "checkpoint-0"

        assert (checkpoint_dir / "A" / "optimizer.pt").exists()
        assert (checkpoint_dir / "B" / "optimizer.pt").exists()
        assert (checkpoint_dir / "A" / "scheduler.pt").exists()
        assert (checkpoint_dir / "B" / "scheduler.pt").exists()

    def test_tokenizer_save_separate(self, causal_model_and_tokenizer, seq2seq_model_and_tokenizer, text_dataset):
        """Test saving separate tokenizers for each model."""
        causal_model, causal_tokenizer = causal_model_and_tokenizer
        seq2seq_model, seq2seq_tokenizer = seq2seq_model_and_tokenizer

        trainer = CycleTrainer(
            args=self.default_args,
            models={"A": causal_model, "B": seq2seq_model},
            tokenizers={"A": causal_tokenizer, "B": seq2seq_tokenizer},
            train_dataset_A=text_dataset,
            train_dataset_B=text_dataset,
        )
        trainer._save_checkpoint(None, None)
        checkpoint_dir = self.output_dir / "checkpoint-0"

        assert (checkpoint_dir / "A" / "tokenizer_config.json").exists()
        assert (checkpoint_dir / "B" / "tokenizer_config.json").exists()
        assert not (checkpoint_dir / "tokenizer_config.json").exists()

    def test_tokenizer_save_shared(self, default_trainer):
        """Test saving shared tokenizer."""
        default_trainer._save_checkpoint(None, None)
        checkpoint_dir = self.output_dir / "checkpoint-0"

        assert (checkpoint_dir / "tokenizer_config.json").exists()
        assert not (checkpoint_dir / "A" / "tokenizer_config.json").exists()
        assert not (checkpoint_dir / "B" / "tokenizer_config.json").exists()

    def test_rng_state_save(self, default_trainer):
        """Test saving RNG states."""
        default_trainer._save_checkpoint(default_trainer.model_A)
        checkpoint_dir = self.output_dir / "checkpoint-0"

        assert (checkpoint_dir / "rng_state.pth").exists()

        rng_state = torch.load(checkpoint_dir / "rng_state.pth", weights_only=False)
        assert "python" in rng_state
        assert "numpy" in rng_state
        assert "cpu" in rng_state
        if torch.cuda.is_available():
            assert "cuda" in rng_state

    def test_trainer_state_save(self, default_trainer):
        """Test saving trainer state."""
        default_trainer._save_checkpoint(default_trainer.model_A)
        checkpoint_dir = self.output_dir / "checkpoint-0"

        assert (checkpoint_dir / "trainer_state.json").exists()

        with open(checkpoint_dir / "trainer_state.json") as f:
            state = json.load(f)
        assert "global_step" in state
        assert "epoch" in state
        assert "stateful_callbacks" in state

    @pytest.mark.parametrize("save_only_model", [True, False], ids=["model_only", "full_checkpoint"])
    def test_save_only_model(self, default_trainer, save_only_model):
        """Test saving only model weights without optimizer/scheduler states."""
        default_trainer.args.save_only_model = save_only_model
        default_trainer._save_checkpoint(default_trainer.model_A)
        checkpoint_dir = self.output_dir / "checkpoint-0"

        assert (checkpoint_dir / "A" / "model.safetensors").exists()
        assert (checkpoint_dir / "B" / "model.safetensors").exists()

        if save_only_model:
            assert not (checkpoint_dir / "A" / "optimizer.pt").exists()
            assert not (checkpoint_dir / "A" / "scheduler.pt").exists()
            assert not (checkpoint_dir / "B" / "optimizer.pt").exists()
            assert not (checkpoint_dir / "B" / "scheduler.pt").exists()
        else:
            assert (checkpoint_dir / "A" / "optimizer.pt").exists()
            assert (checkpoint_dir / "A" / "scheduler.pt").exists()
            assert (checkpoint_dir / "B" / "optimizer.pt").exists()
            assert (checkpoint_dir / "B" / "scheduler.pt").exists()

    def test_save_training_args(self, default_trainer):
        """Test saving training arguments."""
        default_trainer._save_checkpoint(None, None)
        checkpoint_dir = self.output_dir / "checkpoint-0"

        assert (checkpoint_dir / "A" / "training_args.bin").exists()
        assert (checkpoint_dir / "B" / "training_args.bin").exists()
