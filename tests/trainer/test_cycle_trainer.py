import json
import shutil
from pathlib import Path

import pytest
import torch
from peft import PeftModel
from peft.tuners.lora import LoraConfig

from cycleformers import CycleTrainer, CycleTrainingArguments
from cycleformers.cycles import _default_prepare_cycle_inputs, _prepare_causal_skip_cycle_inputs


# TODO: Rename when better defined
@pytest.fixture(name="gen_peft_model")
def fixture_gen_peft_model(request):
    return request.param


class TestCycleTrainerIntegration:
    @pytest.mark.parametrize(
        "peft_config",
        [
            None,
            LoraConfig(r=8, lora_alpha=32),
            {"A": LoraConfig(r=8, lora_alpha=16), "B": LoraConfig(r=16, lora_alpha=32)},
        ],
    )
    def test_use_macct_with_single_model(self, causal_model, causal_tokenizer, peft_config, text_dataset):
        args = CycleTrainingArguments(output_dir="/tmp/cycleformers_test", use_macct=True)
        trainer = CycleTrainer(
            args=args,
            models=causal_model,
            tokenizers=causal_tokenizer,
            train_dataset_A=text_dataset,
            train_dataset_B=text_dataset,
            peft_configs=peft_config,
        )

        assert trainer.is_macct_model
        assert trainer.model_A is trainer.model_B
        assert isinstance(trainer.model_A, PeftModel)
        assert isinstance(trainer.model_B, PeftModel)
        assert trainer.model_A.get_adapter("A") is not None
        assert trainer.model_B.get_adapter("B") is not None

        if peft_config is None:
            assert trainer.model_A.get_adapter("A").r == 8
            assert trainer.model_B.get_adapter("B").r == 8
        elif isinstance(peft_config, LoraConfig):
            assert trainer.model_A.get_adapter("A").r == peft_config.r
            assert trainer.model_B.get_adapter("B").r == peft_config.r
        elif isinstance(peft_config, dict):
            assert trainer.model_A.get_adapter("A").r == peft_config["A"].r
            assert trainer.model_B.get_adapter("B").r == peft_config["B"].r

    @pytest.mark.parametrize(
        "peft_model",
        [
            "gen_peft_model",
            {
                "model": "trl-internal-testing/tiny-LlamaForCausalLM-3.1",
                "preloaded_adapters": None,
                "lora_configs": None,
            },
            {
                "model": "trl-internal-testing/tiny-LlamaForCausalLM-3.1",
                "preloaded_adapters": ["A"],
                "lora_configs": None,
            },
            {
                "model": "trl-internal-testing/tiny-LlamaForCausalLM-3.1",
                "preloaded_adapters": ["A", "B"],
                "lora_configs": None,
            },
        ],
    )
    def test_is_macct_with_single_peft_model(self, model, tokenizer, preloaded_adapters, lora_configs, text_dataset):
        args = CycleTrainingArguments(output_dir="/tmp/cycleformers_test", use_macct=True)
        trainer = CycleTrainer(
            args=args,
            models=model,
            tokenizers=tokenizer,
            train_dataset_A=text_dataset,
            train_dataset_B=text_dataset,
            peft_configs=lora_configs,
        )


class TestSetCycleInputsFn:
    def test_set_custom_fn(self, causal_model, causal_tokenizer, text_dataset):
        def test_fn(self, *args, **kwargs):
            return args, kwargs

        trainer = CycleTrainer(
            args=CycleTrainingArguments(output_dir="/tmp/cycleformers_test"),
            models={"A": causal_model, "B": causal_model},
            tokenizers=causal_tokenizer,
            train_dataset_A=text_dataset,
            train_dataset_B=text_dataset,
        )
        trainer.set_cycle_inputs_fn(test_fn)
        assert trainer._prepare_cycle_inputs.__func__ == test_fn

    def test_set_default_causal_skip(self, causal_model, causal_tokenizer, text_dataset):
        """Both causal models with matching tokenizers"""
        trainer = CycleTrainer(
            args=CycleTrainingArguments(output_dir="/tmp/cycleformers_test"),
            models={"A": causal_model, "B": causal_model},
            tokenizers=causal_tokenizer,
            train_dataset_A=text_dataset,
            train_dataset_B=text_dataset,
        )
        trainer.set_cycle_inputs_fn()
        assert trainer._prepare_cycle_inputs.__func__ == _prepare_causal_skip_cycle_inputs

    def test_set_default_default(self, causal_model, seq2seq_model, causal_tokenizer, seq2seq_tokenizer, text_dataset):
        """Any other case"""
        trainer = CycleTrainer(
            args=CycleTrainingArguments(output_dir="/tmp/cycleformers_test"),
            models={"A": seq2seq_model, "B": causal_model},
            tokenizers={"A": seq2seq_tokenizer, "B": causal_tokenizer},
            train_dataset_A=text_dataset,
            train_dataset_B=text_dataset,
        )
        trainer.set_cycle_inputs_fn()
        assert trainer._prepare_cycle_inputs.__func__ == _default_prepare_cycle_inputs


class TestSaveCheckpoint:
    @pytest.fixture(autouse=True)
    def setup(self, causal_model, causal_tokenizer, text_dataset):
        """Setup for each test"""
        self.save_dir = Path("/tmp/cycleformers_test")
        if self.save_dir.exists():
            shutil.rmtree(self.save_dir)
        self.save_dir.mkdir(parents=True)

        self.args = CycleTrainingArguments(
            output_dir=str(self.save_dir),
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=1,
            save_strategy="steps",
            save_steps=len(text_dataset) - 1,
        )

        self.trainer = CycleTrainer(
            args=self.args,
            models={"A": causal_model, "B": causal_model},
            tokenizers=causal_tokenizer,
            train_dataset_A=text_dataset,
            train_dataset_B=text_dataset,
        )
        yield

        if self.save_dir.exists():
            shutil.rmtree(self.save_dir)

    def test_lora_adapters_save(self, peft_causal_model, causal_tokenizer, text_dataset):
        """Test that LoRA adapters are saved correctly"""
        self.args.use_macct = True  # Needed to be activated
        trainer = CycleTrainer(
            args=self.args,
            models=peft_causal_model,
            tokenizers=causal_tokenizer,
            train_dataset_A=text_dataset,
            train_dataset_B=text_dataset,
        )
        trainer._save_checkpoint(None, None)
        checkpoint_dir = self.save_dir / "checkpoint-0"

        assert (checkpoint_dir / "A" / "adapter_model.safetensors").exists()
        assert (checkpoint_dir / "B" / "adapter_model.safetensors").exists()
        assert (checkpoint_dir / "A" / "adapter_config.json").exists()
        assert (checkpoint_dir / "B" / "adapter_config.json").exists()

    def test_model_save(self):
        """Test that models are saved correctly"""
        self.trainer._save_checkpoint(None, None)
        checkpoint_dir = self.save_dir / "checkpoint-0"

        assert (checkpoint_dir / "A" / "model.safetensors").exists()
        assert (checkpoint_dir / "B" / "model.safetensors").exists()

    def test_optimizer_scheduler_save(self):
        """Test that optimizers and schedulers are saved correctly"""
        self.trainer._save_checkpoint(None, None)
        checkpoint_dir = self.save_dir / "checkpoint-0"

        assert (checkpoint_dir / "A" / "optimizer.pt").exists()
        assert (checkpoint_dir / "B" / "optimizer.pt").exists()
        assert (checkpoint_dir / "A" / "scheduler.pt").exists()
        assert (checkpoint_dir / "B" / "scheduler.pt").exists()

    def test_tokenizer_save_separate(self, causal_model, causal_tokenizer, seq2seq_tokenizer, text_dataset):
        """Test tokenizers are saved separately when different"""
        trainer = CycleTrainer(
            args=self.args,
            models={"A": causal_model, "B": causal_model},
            tokenizers={"A": causal_tokenizer, "B": seq2seq_tokenizer},
            train_dataset_A=text_dataset,
            train_dataset_B=text_dataset,
        )
        trainer._save_checkpoint(None, None)
        checkpoint_dir = self.save_dir / "checkpoint-0"

        assert (checkpoint_dir / "A" / "tokenizer_config.json").exists()
        assert (checkpoint_dir / "B" / "tokenizer_config.json").exists()
        assert not (checkpoint_dir / "tokenizer_config.json").exists()

    def test_tokenizer_save_shared(self):
        """Test shared tokenizer is saved in root directory"""
        self.trainer._save_checkpoint(None, None)
        checkpoint_dir = self.save_dir / "checkpoint-0"

        assert (checkpoint_dir / "tokenizer_config.json").exists()
        assert not (checkpoint_dir / "A" / "tokenizer_config.json").exists()
        assert not (checkpoint_dir / "B" / "tokenizer_config.json").exists()

    def test_rng_state_save(self):
        """Test that RNG state is saved correctly"""
        self.trainer._save_checkpoint(self.trainer.model_A)
        checkpoint_dir = self.save_dir / "checkpoint-0"

        assert (checkpoint_dir / "rng_state.pth").exists()

        rng_state = torch.load(checkpoint_dir / "rng_state.pth", weights_only=False)
        assert "python" in rng_state
        assert "numpy" in rng_state
        assert "cpu" in rng_state
        if torch.cuda.is_available():
            assert "cuda" in rng_state

    def test_trainer_state_save(self):
        """Test that trainer state is saved correctly"""
        self.trainer._save_checkpoint(self.trainer.model_A)
        checkpoint_dir = self.save_dir / "checkpoint-0"

        assert (checkpoint_dir / "trainer_state.json").exists()

        with open(checkpoint_dir / "trainer_state.json") as f:
            state = json.load(f)
        assert "global_step" in state
        assert "epoch" in state
        assert "stateful_callbacks" in state

    def test_save_only_model(self):
        """Test save_only_model flag"""
        try:
            self.trainer.args.save_only_model = True
            self.trainer._save_checkpoint(self.trainer.model_A)
            checkpoint_dir = self.save_dir / "checkpoint-0"

            assert (checkpoint_dir / "A" / "model.safetensors").exists()
            assert (checkpoint_dir / "B" / "model.safetensors").exists()

            assert not (checkpoint_dir / "A" / "optimizer.pt").exists()
            assert not (checkpoint_dir / "A" / "scheduler.pt").exists()
            assert not (checkpoint_dir / "B" / "optimizer.pt").exists()
            assert not (checkpoint_dir / "B" / "scheduler.pt").exists()
        finally:
            self.trainer.args.save_only_model = False

    def test_save_training_args(self):
        """Test that training arguments are saved correctly"""
        self.trainer._save_checkpoint(None, None)
        checkpoint_dir = self.save_dir / "checkpoint-0"

        assert (checkpoint_dir / "A" / "training_args.bin").exists()
        assert (checkpoint_dir / "B" / "training_args.bin").exists()
