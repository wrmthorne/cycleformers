# Usage

## Basic Training

The `CycleTrainer` extends the 🤗 Transformers trainer to support cycle training. It handles both standard two-model training and memory-efficient adapter-based training.

### Standard Training

Basic example with two identical models:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from cycleformers import CycleTrainer, CycleTrainingArguments
from datasets import load_dataset

# Load models and tokenizers
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Load datasets (e.g., WMT14 English-German)
dataset = load_dataset("wmt14", "de-en")
dataset_en = dataset["train"].select_columns(["translation"]).map(lambda x: {"text": x["translation"]["en"]})
dataset_de = dataset["train"].select_columns(["translation"]).map(lambda x: {"text": x["translation"]["de"]})

# Setup trainer
args = CycleTrainingArguments(
    output_dir="gpt2-cct",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=1000,
)

trainer = CycleTrainer(
    args=args,
    models=model,  # Will be duplicated internally
    tokenizers=tokenizer,
    train_dataset_A=dataset_en,
    train_dataset_B=dataset_de,
)

# Start training
trainer.train()
```

### Different Models

Using two different models (must be same architecture type):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load different models
model_A = AutoModelForCausalLM.from_pretrained("gpt2")
model_B = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
tokenizer_A = AutoTokenizer.from_pretrained("gpt2")
tokenizer_B = AutoTokenizer.from_pretrained("facebook/opt-125m")

trainer = CycleTrainer(
    args=args,
    models={"A": model_A, "B": model_B},
    tokenizers={"A": tokenizer_A, "B": tokenizer_B},
    train_dataset_A=dataset_en,
    train_dataset_B=dataset_de,
)
```

## Memory-Efficient Training (MACCT)

Train a single larger model with two PEFT adapters, using ~7.5x less memory than standard training:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from cycleformers import CycleTrainer, CycleTrainingArguments

# Load base model
model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

# Configure LoRA adapters
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)

# Setup trainer with MACCT enabled
args = CycleTrainingArguments(
    output_dir="opt-macct",
    use_macct=True,  # Enable adapter-based training
    per_device_train_batch_size=4,
    num_train_epochs=3,
)

trainer = CycleTrainer(
    args=args,
    models=model,
    tokenizers=tokenizer,
    train_dataset_A=dataset_en,
    train_dataset_B=dataset_de,
    peft_configs=peft_config,  # Will be used for both adapters
)

trainer.train()
```

## Using Task Processors

For standard tasks like translation or NER, use the built-in processors:

```python
from cycleformers import AutoProcessor
from cycleformers.task_processors.translation import TranslationProcessorConfig

# Load and process WMT14 dataset
config = TranslationProcessorConfig(
    dataset_name="wmt14",
    dataset_config_name="de-en",
    source_lang="en",
    target_lang="de",
)

processor = AutoProcessor.from_config(config)
dataset_A, dataset_B = processor.process()

# Use processed datasets in trainer
trainer = CycleTrainer(
    args=args,
    models=model,
    tokenizers=tokenizer,
    train_dataset_A=dataset_A,
    train_dataset_B=dataset_B,
)
```