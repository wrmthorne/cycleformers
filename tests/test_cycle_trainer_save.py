import os
import shutil
import pytest
import torch
import json
from pathlib import Path
from cycleformers.trainer import CycleTrainer, CycleTrainingArguments

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
            save_steps=len(text_dataset) - 1
        )

        self.trainer = CycleTrainer(
            args=self.args,
            models={'A': causal_model, 'B': causal_model},
            tokenizers=causal_tokenizer,
            train_dataset_A=text_dataset,
            train_dataset_B=text_dataset
        )
        yield

        if self.save_dir.exists():
            shutil.rmtree(self.save_dir)


    def test_lora_adapters_save(self, peft_causal_model, causal_tokenizer, text_dataset):
        """Test that LoRA adapters are saved correctly"""               
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
            models={'A': causal_model, 'B': causal_model},
            tokenizers={'A': causal_tokenizer, 'B': seq2seq_tokenizer},
            train_dataset_A=text_dataset,
            train_dataset_B=text_dataset
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

        rng_state = torch.load(checkpoint_dir / "rng_state.pth")
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


    @pytest.mark.skip(reason="Will become relevant when separate model configs are implemented")
    def test_save_training_args(self):
        """Test that training arguments are saved correctly"""
        self.trainer._save_checkpoint(None, None)
        checkpoint_dir = self.save_dir / "checkpoint-0"

        assert (checkpoint_dir / "training_args.bin").exists()