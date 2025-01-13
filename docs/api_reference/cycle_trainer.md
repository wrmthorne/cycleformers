# CycleTrainer

The `CycleTrainer` class extends HuggingFace's `Trainer` to support cycle training between two language models or two adapters of a single model. It handles both standard cycle training (two separate models) and memory-efficient cycle training (single model with PEFT adapters).

::: cycleformers.cycle_trainer.CycleTrainer

## Key Features

- Supports both standard two-model cycle training and memory-efficient adapter-based training (MACCT)
- Compatible with causal language models and encoder-decoder models
- Integrates with HuggingFace's ecosystem (datasets, tokenizers, etc.)
- Customizable cycle input preparation and evaluation metrics
- Built-in support for PEFT adapters

## Methods

### train

::: cycleformers.cycle_trainer.CycleTrainer.train

The training process alternates between:
1. Model A generating text from Model B's training data and training to reconstruct B's input
2. Model B generating text from Model A's training data and training to reconstruct A's input

### evaluate

::: cycleformers.cycle_trainer.CycleTrainer.evaluate

### _cycle_step

::: cycleformers.cycle_trainer.CycleTrainer._cycle_step

## Example Usage

```python
from cycleformers import CycleTrainer, CycleTrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize models and tokenizers
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Setup training arguments
args = CycleTrainingArguments(
    output_dir="gpt2-cycle-training",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=1000,
    logging_steps=100
)

# Create trainer
trainer = CycleTrainer(
    args=args,
    models=model,  # Will be duplicated for standard training
    tokenizers=tokenizer,
    train_dataset_A=dataset_A,
    train_dataset_B=dataset_B,
    eval_dataset_A=eval_dataset_A,
    eval_dataset_B=eval_dataset_B
)

# Start training
trainer.train()
```

For memory-efficient training with PEFT adapters:

```python
from peft import LoraConfig

# Create PEFT config
peft_config = LoraConfig(r=8, lora_alpha=32)

# Setup training arguments with MACCT enabled
args = CycleTrainingArguments(
    output_dir="gpt2-macct",
    use_macct=True,
    per_device_train_batch_size=4
)

# Create trainer with PEFT config
trainer = CycleTrainer(
    args=args,
    models=model,
    tokenizers=tokenizer,
    train_dataset_A=dataset_A,
    train_dataset_B=dataset_B,
    peft_configs=peft_config  # Will be used for both adapters
)

trainer.train()
```