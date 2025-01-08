import json
import random
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from datasets import Dataset

from cycleformers import CycleTrainer, CycleTrainingArguments
from cycleformers.cycles import _default_prepare_cycle_inputs, _prepare_causal_skip_cycle_inputs
from cycleformers.exceptions import InvalidCycleKeyError, MACCTModelError, MissingModelError
from cycleformers.import_utils import is_peft_available
from tests.testing_utils.custom_marks import requires_peft


if is_peft_available():
    from peft import PeftConfig, PeftModel
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
    def default_trainer(self, any_model_and_tokenizer_pairs, text_dataset):
        """Create a default trainer instance with common configuration."""
        model_and_tokenizer_A, model_and_tokenizer_B = any_model_and_tokenizer_pairs
        model_A, tokenizer_A = model_and_tokenizer_A
        model_B, tokenizer_B = model_and_tokenizer_B

        return CycleTrainer(
            args=self.default_args,
            models={"A": model_A, "B": model_B},
            tokenizers={"A": tokenizer_A, "B": tokenizer_B},
            train_dataset_A=text_dataset,
            train_dataset_B=text_dataset,
        )

    @pytest.fixture
    def same_model_trainer(self, same_model_and_tokenizer_pairs, text_dataset):
        """Create a trainer instance with two identical models."""
        model_and_tokenizer_A, model_and_tokenizer_B = same_model_and_tokenizer_pairs
        model_A, tokenizer_A = model_and_tokenizer_A
        model_B, _ = model_and_tokenizer_B

        return CycleTrainer(
            args=self.default_args,
            models={"A": model_A, "B": model_B},
            tokenizers=tokenizer_A,
            train_dataset_A=text_dataset,
            train_dataset_B=text_dataset,
        )

    @pytest.fixture
    def macct_trainer(self, any_peft_model_and_tokenizer, text_dataset):
        """Create a trainer instance with MACCT mode."""
        model, tokenizer = any_peft_model_and_tokenizer
        args = self.default_args
        args.use_macct = True

        return CycleTrainer(
            args=args,
            models=model,
            tokenizers=tokenizer,
            train_dataset_A=text_dataset,
            train_dataset_B=text_dataset,
        )

    def assert_checkpoint_exists(self, checkpoint_dir: Path, is_macct: bool = False, save_only_model: bool = False):
        """Assert that a checkpoint directory contains all expected files."""
        if is_macct:
            # MACCT model saves adapters and shared tokenizer
            assert (checkpoint_dir / "A" / "adapter_model.safetensors").exists()
            assert (checkpoint_dir / "B" / "adapter_model.safetensors").exists()
            assert (checkpoint_dir / "A" / "adapter_config.json").exists()
            assert (checkpoint_dir / "B" / "adapter_config.json").exists()
            assert (checkpoint_dir / "tokenizer_config.json").exists()  # Shared tokenizer
        else:
            # Regular model saves full models and tokenizers
            assert (checkpoint_dir / "A" / "model.safetensors").exists()
            assert (checkpoint_dir / "B" / "model.safetensors").exists()
            assert (checkpoint_dir / "tokenizer_config.json").exists()  # FIXME: tokenizer may be shared or separate

        # Optimizer/scheduler files based on save_only_model
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

        assert (checkpoint_dir / "trainer_state.json").exists()

    def assert_trainer_model_classes(
        self, trainer, is_macct: bool = False, peft_configs: PeftConfig | dict[str, PeftConfig] | None = None
    ):
        """Assert that the trainer models are of the correct class and configuration."""
        # Verify MACCT setup
        if is_macct:
            assert trainer.is_macct_model
            assert trainer.model_A is trainer.model_B
            assert isinstance(trainer.model_A, PeftModel)
            assert isinstance(trainer.model_B, PeftModel)
            assert trainer.model_A.peft_config["A"] is not None
            assert trainer.model_B.peft_config["B"] is not None
        else:
            assert not trainer.is_macct_model

        # Verify PEFT configurations
        if peft_configs is not None:
            # Single config for both models
            if isinstance(peft_configs, LoraConfig):
                assert isinstance(trainer.model_A, PeftModel)
                assert isinstance(trainer.model_B, PeftModel)
                assert trainer.model_A.peft_config["A"].r == peft_configs.r
                assert trainer.model_B.peft_config["B"].r == peft_configs.r

            # Dict config with different settings per model
            elif isinstance(peft_configs, dict):
                # Check each model's configuration
                if "A" in peft_configs:
                    assert isinstance(trainer.model_A, PeftModel)
                    assert trainer.model_A.peft_config["A"].r == peft_configs["A"].r

                if "B" in peft_configs:
                    assert isinstance(trainer.model_B, PeftModel)
                    assert trainer.model_B.peft_config["B"].r == peft_configs["B"].r
        else:
            # No PEFT configs, should be regular models unless in MACCT mode
            if not is_macct:
                assert not isinstance(trainer.model_A, PeftModel)
                assert not isinstance(trainer.model_B, PeftModel)


@requires_peft
class TestModelIngest(TrainerTestMixin):
    """Tests for model ingestion logic in CycleTrainer.

    These tests verify the different ways models can be provided to CycleTrainer:
    - String model name
    - Single model (with/without PEFT)
    - Dictionary of models
    - PeftModel with adapters
    - Various combinations of PEFT configs
    """

    @pytest.mark.parametrize("use_macct", [True, False])
    def test_single_model_string(self, model_registry, text_dataset, use_macct):
        """Test providing a single model as a string."""
        model_spec = random.choice(model_registry.get_matching_models())
        args = self.default_args
        args.use_macct = use_macct

        trainer = CycleTrainer(
            args=args,
            models=model_spec.repo_id,
            train_dataset_A=text_dataset,
            train_dataset_B=text_dataset,
            peft_configs=LoraConfig(r=8, lora_alpha=16),  # TODO: Test when config is not provided
        )

        if use_macct:
            assert trainer.model_A.base_model is trainer.model_B.base_model
            assert trainer.is_macct_model
        else:
            assert trainer.model_A is not trainer.model_B
            assert not trainer.is_macct_model

    @pytest.mark.parametrize("use_macct", [True, False])
    def test_single_model_no_peft(self, causal_model_and_tokenizer, text_dataset, use_macct):
        """Test providing a single model without PEFT configuration."""
        model, tokenizer = causal_model_and_tokenizer
        args = self.default_args
        args.use_macct = use_macct

        if use_macct:
            expected_error = MACCTModelError
        else:
            expected_error = MissingModelError

        with pytest.raises(expected_error):
            CycleTrainer(
                args=args,
                models=model,
                tokenizers=tokenizer,
                train_dataset_A=text_dataset,
                train_dataset_B=text_dataset,
            )

    @pytest.mark.parametrize("use_macct", [True, False])
    @pytest.mark.parametrize(
        "peft_config",
        [
            LoraConfig(r=8, lora_alpha=32),  # Single config
            {"A": LoraConfig(r=8, lora_alpha=16)},  # Single config for A
            {"A": LoraConfig(r=8, lora_alpha=16), "B": LoraConfig(r=16, lora_alpha=32)},  # Dict config
        ],
    )
    def test_single_model_with_peft(self, random_model_and_tokenizer, text_dataset, use_macct, peft_config):
        """Test providing a single model with PEFT configuration."""
        model, tokenizer = random_model_and_tokenizer
        args = self.default_args
        args.use_macct = use_macct

        if not use_macct:
            expected_error = MissingModelError
        elif use_macct and isinstance(peft_config, dict) and len(peft_config) == 1:
            expected_error = MACCTModelError
        else:
            expected_error = None

        if expected_error is not None:
            with pytest.raises(expected_error):
                CycleTrainer(
                    args=args,
                    models=model,
                    tokenizers=tokenizer,
                    train_dataset_A=text_dataset,
                    train_dataset_B=text_dataset,
                    peft_configs=peft_config,
                )
        else:
            trainer = CycleTrainer(
                args=args,
                models=model,
                tokenizers=tokenizer,
                train_dataset_A=text_dataset,
                train_dataset_B=text_dataset,
                peft_configs=peft_config,
            )

            self.assert_trainer_model_classes(trainer, is_macct=True, peft_configs=peft_config)

    def test_dict_model_no_peft(self, any_model_and_tokenizer_pairs, text_dataset):
        """Test providing a model dictionary without PEFT configuration."""
        model_and_tokenizer_A, model_and_tokenizer_B = any_model_and_tokenizer_pairs
        model_A, tokenizer_A = model_and_tokenizer_A
        model_B, tokenizer_B = model_and_tokenizer_B
        args = self.default_args

        trainer = CycleTrainer(
            args=args,
            models={"A": model_A, "B": model_B},
            tokenizers={"A": tokenizer_A, "B": tokenizer_B},
            train_dataset_A=text_dataset,
            train_dataset_B=text_dataset,
        )

        # Verify model setup
        self.assert_trainer_model_classes(trainer, is_macct=False, peft_configs=None)

        # Verify tokenizer assignments
        assert trainer.model_A is model_A
        assert trainer.model_B is model_B
        assert trainer.tokenizer_A is tokenizer_A
        assert trainer.tokenizer_B is tokenizer_B

    @pytest.mark.parametrize(
        "peft_config",
        [
            LoraConfig(r=8, lora_alpha=32),  # Single config
            {"A": LoraConfig(r=8, lora_alpha=16)},  # Single config for A
            {"A": LoraConfig(r=8, lora_alpha=16), "B": LoraConfig(r=16, lora_alpha=32)},  # Dict config
        ],
    )
    def test_dict_model_with_peft(self, any_model_and_tokenizer_pairs, text_dataset, peft_config):
        """Test providing a model dictionary with PEFT configuration."""
        model_and_tokenizer_A, model_and_tokenizer_B = any_model_and_tokenizer_pairs
        model_A, tokenizer_A = model_and_tokenizer_A
        model_B, tokenizer_B = model_and_tokenizer_B
        args = self.default_args

        trainer = CycleTrainer(
            args=args,
            models={"A": model_A, "B": model_B},
            tokenizers={"A": tokenizer_A, "B": tokenizer_B},
            train_dataset_A=text_dataset,
            train_dataset_B=text_dataset,
            peft_configs=peft_config,
        )

        # Verify model setup
        self.assert_trainer_model_classes(trainer, is_macct=False, peft_configs=peft_config)

        # Verify tokenizer assignments
        assert trainer.tokenizer_A is tokenizer_A
        assert trainer.tokenizer_B is tokenizer_B

    def test_dict_model_with_macct(self, any_model_and_tokenizer_pairs, text_dataset):
        """Test providing a model dictionary with MACCT mode."""
        model_and_tokenizer_A, model_and_tokenizer_B = any_model_and_tokenizer_pairs
        model_A, tokenizer_A = model_and_tokenizer_A
        model_B, tokenizer_B = model_and_tokenizer_B
        args = self.default_args
        args.use_macct = True

        # Too many models
        with pytest.raises(MACCTModelError, match="Cannot use dictionary of models in MACCT mode"):
            CycleTrainer(
                args=args,
                models={"A": model_A, "B": model_B},
                tokenizers={"A": tokenizer_A, "B": tokenizer_B},
                train_dataset_A=text_dataset,
                train_dataset_B=text_dataset,
            )

    def test_invalid_model_dict(self, any_model_and_tokenizer_pairs, text_dataset):
        """Test providing an invalid model dictionary."""
        model_and_tokenizer_A, model_and_tokenizer_B = any_model_and_tokenizer_pairs
        model_A, tokenizer_A = model_and_tokenizer_A
        model_B, tokenizer_B = model_and_tokenizer_B

        # Invalid keys
        with pytest.raises(InvalidCycleKeyError):
            CycleTrainer(
                args=self.default_args,
                models={"A": model_A, "C": model_B},  # Invalid key 'C'
                tokenizers={"A": tokenizer_A, "C": tokenizer_B},
                train_dataset_A=text_dataset,
                train_dataset_B=text_dataset,
            )

        # Missing required key
        with pytest.raises(InvalidCycleKeyError):
            CycleTrainer(
                args=self.default_args,
                models={"A": model_A},  # Missing key 'B'
                tokenizers={"A": tokenizer_A, "B": tokenizer_B},
                train_dataset_A=text_dataset,
                train_dataset_B=text_dataset,
            )

        # Empty dict
        with pytest.raises(MissingModelError, match="didn't receive any models"):
            CycleTrainer(
                args=self.default_args,
                models={},
                tokenizers=tokenizer_A,
                train_dataset_A=text_dataset,
                train_dataset_B=text_dataset,
            )

    @pytest.mark.parametrize("missing_adapter", ["A", "B"])
    def test_peft_model_missing_adapter(self, any_peft_model_and_tokenizer, text_dataset, missing_adapter):
        """Test providing a PeftModel with missing adapters."""
        model, tokenizer = any_peft_model_and_tokenizer
        args = self.default_args
        args.use_macct = True

        # Remove one adapter
        if missing_adapter in model.peft_config:
            model.delete_adapter(missing_adapter)

        # Should fail without peft_configs
        with pytest.raises(MACCTModelError, match="no PeftConfig was provided"):
            CycleTrainer(
                args=args,
                models=model,
                tokenizers=tokenizer,
                train_dataset_A=text_dataset,
                train_dataset_B=text_dataset,
            )

        # Should work with peft_configs
        peft_config = LoraConfig(r=8, lora_alpha=32)
        trainer = CycleTrainer(
            args=args,
            models=model,
            tokenizers=tokenizer,
            train_dataset_A=text_dataset,
            train_dataset_B=text_dataset,
            peft_configs={missing_adapter: peft_config},
        )

        # Verify model setup
        self.assert_trainer_model_classes(trainer, is_macct=True, peft_configs={missing_adapter: peft_config})

    def test_peft_model_no_macct(self, any_peft_model_and_tokenizer, text_dataset):
        """Test providing a PeftModel without MACCT mode."""
        model, tokenizer = any_peft_model_and_tokenizer
        args = self.default_args
        args.use_macct = False

        # Should raise error - can't use single model in non-MACCT mode
        with pytest.raises(MissingModelError):
            CycleTrainer(
                args=args,
                models=model,
                tokenizers=tokenizer,
                train_dataset_A=text_dataset,
                train_dataset_B=text_dataset,
            )


class TestSetCycleInputsFn(TrainerTestMixin):
    """Tests for cycle input preparation functionality."""

    def test_set_custom_fn(self, default_trainer):
        """Test setting a custom cycle inputs preparation function."""

        def test_fn(*args, **kwargs):
            return args, kwargs

        default_trainer.set_cycle_inputs_fn(test_fn)
        assert default_trainer._prepare_cycle_inputs.__func__ == test_fn

    def test_set_default_causal_skip(self, causal_model_and_tokenizer, text_dataset):
        """Test default cycle inputs function for causal models."""
        model, tokenizer = causal_model_and_tokenizer
        trainer = CycleTrainer(
            args=self.default_args,
            models={"A": model, "B": deepcopy(model)},
            tokenizers=tokenizer,
            train_dataset_A=text_dataset,
            train_dataset_B=text_dataset,
        )
        trainer.set_cycle_inputs_fn()
        assert trainer._prepare_cycle_inputs.__func__ == _prepare_causal_skip_cycle_inputs

    def test_set_default_prepare_cycle_inputs(self, diff_model_and_tokenizer_pairs, text_dataset):
        """Test default cycle inputs function for mixed model types."""
        model_and_tokenizer_A, model_and_tokenizer_B = diff_model_and_tokenizer_pairs
        model_A, tokenizer_A = model_and_tokenizer_A
        model_B, tokenizer_B = model_and_tokenizer_B

        trainer = CycleTrainer(
            args=self.default_args,
            models={"A": model_A, "B": model_B},
            tokenizers={"A": tokenizer_A, "B": tokenizer_B},
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

    def test_adapter_save(self, macct_trainer):
        """Test saving model weights in MACCT mode."""
        macct_trainer._save_checkpoint(None, None)
        checkpoint_dir = self.output_dir / "checkpoint-0"

        assert (checkpoint_dir / "A" / "adapter_model.safetensors").exists()
        assert (checkpoint_dir / "B" / "adapter_model.safetensors").exists()
        assert (checkpoint_dir / "A" / "adapter_config.json").exists()
        assert (checkpoint_dir / "B" / "adapter_config.json").exists()

    def test_optimizer_scheduler_save(self, default_trainer):
        """Test saving optimizer and scheduler states."""
        default_trainer._save_checkpoint(None, None)
        checkpoint_dir = self.output_dir / "checkpoint-0"

        assert (checkpoint_dir / "A" / "optimizer.pt").exists()
        assert (checkpoint_dir / "B" / "optimizer.pt").exists()
        assert (checkpoint_dir / "A" / "scheduler.pt").exists()
        assert (checkpoint_dir / "B" / "scheduler.pt").exists()

    def test_tokenizer_save_separate(self, diff_model_and_tokenizer_pairs, text_dataset):
        """Test saving separate tokenizers for each model."""
        model_and_tokenizer_A, model_and_tokenizer_B = diff_model_and_tokenizer_pairs
        model_A, tokenizer_A = model_and_tokenizer_A
        model_B, tokenizer_B = model_and_tokenizer_B

        trainer = CycleTrainer(
            args=self.default_args,
            models={"A": model_A, "B": model_B},
            tokenizers={"A": tokenizer_A, "B": tokenizer_B},
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
