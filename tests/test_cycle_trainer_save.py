import os
import shutil
import pytest
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from cycleformers.trainer import CycleTrainer, CycleTrainingArguments
from peft import LoraConfig, get_peft_model

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
            train_dataset_B=text_dataset,
            eval_dataset_A=text_dataset, 
            eval_dataset_B=text_dataset
        )

        yield

        # Cleanup
        if self.save_dir.exists():
            shutil.rmtree(self.save_dir)

    def test_base_models_save_in_subdirs(self):
        """Test that models A and B are saved in their respective subdirectories"""
        self.trainer._save_checkpoint(None, None)
        checkpoint_dir = self.save_dir / "checkpoint-0"
        
        # Check model directories exist
        assert (checkpoint_dir / "A").exists()
        assert (checkpoint_dir / "B").exists()
        
        # Check model files exist in correct locations
        assert (checkpoint_dir / "A" / "pytorch_model.bin").exists()
        assert (checkpoint_dir / "B" / "pytorch_model.bin").exists()
        assert (checkpoint_dir / "A" / "config.json").exists()
        assert (checkpoint_dir / "B" / "config.json").exists()

    def test_lora_adapters_save(self, peft_causal_model):
        """Test that LoRA adapters are saved correctly"""               
        # Update trainer with PEFT model
        self.trainer.model_A = self.trainer.model_B = peft_causal_model
        
        # Save checkpoint
        self.trainer._save_checkpoint(None, None)
        checkpoint_dir = self.save_dir / "checkpoint-0"
        
        # Check adapter files exist
        assert (checkpoint_dir / "A" / "adapter_model.bin").exists()
        assert (checkpoint_dir / "B" / "adapter_model.bin").exists()
        assert (checkpoint_dir / "A" / "adapter_config.json").exists()
        assert (checkpoint_dir / "B" / "adapter_config.json").exists()

    def test_optimizer_scheduler_save(self):
        """Test that optimizers and schedulers are saved correctly"""
        self.trainer._save_checkpoint(self.trainer.model_A)
        checkpoint_dir = self.save_dir / "checkpoint-0"
        
        # Check optimizer files
        assert (checkpoint_dir / "A" / "optimizer.pt").exists()
        assert (checkpoint_dir / "B" / "optimizer.pt").exists()
        
        # Check scheduler files
        assert (checkpoint_dir / "A" / "scheduler.pt").exists()
        assert (checkpoint_dir / "B" / "scheduler.pt").exists()
        
        # Verify optimizer state can be loaded
        optimizer_state_A = torch.load(checkpoint_dir / "A" / "optimizer.pt")
        optimizer_state_B = torch.load(checkpoint_dir / "B" / "optimizer.pt")
        assert "state" in optimizer_state_A
        assert "state" in optimizer_state_B

    def test_tokenizer_save_separate(self):
        """Test tokenizers are saved separately when different"""
        # Create different tokenizer for model B
        tokenizer_B = AutoTokenizer.from_pretrained("gpt2")
        self.trainer.tokenizer_B = tokenizer_B
        
        self.trainer._save_checkpoint(self.trainer.model_A)
        checkpoint_dir = self.save_dir / "checkpoint-0"
        
        # Check tokenizer files exist in separate directories
        assert (checkpoint_dir / "A" / "tokenizer_config.json").exists()
        assert (checkpoint_dir / "B" / "tokenizer_config.json").exists()

    def test_tokenizer_save_shared(self):
        """Test shared tokenizer is saved in root directory"""
        self.trainer._save_checkpoint(self.trainer.model_A)
        checkpoint_dir = self.save_dir / "checkpoint-0"
        
        # Check tokenizer files exist in root directory
        assert (checkpoint_dir / "tokenizer_config.json").exists()
        assert not (checkpoint_dir / "A" / "tokenizer_config.json").exists()
        assert not (checkpoint_dir / "B" / "tokenizer_config.json").exists()

    def test_rng_state_save(self):
        """Test that RNG state is saved correctly"""
        self.trainer._save_checkpoint(self.trainer.model_A)
        checkpoint_dir = self.save_dir / "checkpoint-0"
        
        # Check RNG state file exists
        assert (checkpoint_dir / "rng_state.pth").exists()
        
        # Load and verify RNG state
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
        
        # Check trainer state file exists
        assert (checkpoint_dir / "trainer_state.json").exists()
        
        # Load and verify trainer state
        with open(checkpoint_dir / "trainer_state.json") as f:
            state = json.load(f)
        assert "global_step" in state
        assert "epoch" in state
        assert "stateful_callbacks" in state

    def test_save_safetensors(self):
        """Test saving in safetensors format"""
        self.trainer.args.save_safetensors = True
        self.trainer._save_checkpoint(self.trainer.model_A)
        checkpoint_dir = self.save_dir / "checkpoint-0"
        
        # Check safetensors files exist instead of pytorch_model.bin
        assert (checkpoint_dir / "A" / "model.safetensors").exists()
        assert (checkpoint_dir / "B" / "model.safetensors").exists()
        assert not (checkpoint_dir / "A" / "pytorch_model.bin").exists()
        assert not (checkpoint_dir / "B" / "pytorch_model.bin").exists()

    def test_save_only_model(self):
        """Test save_only_model flag"""
        self.trainer.args.save_only_model = True
        self.trainer._save_checkpoint(self.trainer.model_A)
        checkpoint_dir = self.save_dir / "checkpoint-0"
        
        # Check model files exist
        assert (checkpoint_dir / "A" / "pytorch_model.bin").exists()
        assert (checkpoint_dir / "B" / "pytorch_model.bin").exists()
        
        # Check optimizer/scheduler files don't exist
        assert not (checkpoint_dir / "A" / "optimizer.pt").exists()
        assert not (checkpoint_dir / "A" / "scheduler.pt").exists()
        assert not (checkpoint_dir / "B" / "optimizer.pt").exists()
        assert not (checkpoint_dir / "B" / "scheduler.pt").exists()