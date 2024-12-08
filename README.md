# Cycleformers

<div align="center">

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
<!-- ![Coverage](.github/badges/coverage.svg) -->
<!-- [![Build Status](https://github.com/wrmthorne/cycleformers/workflows/CI-Pipeline/badge.svg)](https://github.com/wrmthorne/cycleformers/actions) -->

</div>

A Python library for efficient cycle-consistency training of transformer models. Cycleformers simplifies iterative back-translation with support for both causal and seq2seq architectures. We also implement Multi-Adapter Cycle-Consistency Training (MACCT), enabling training of LoRA adapters on a frozen base model for `7.5x` larger model capacity for the same memory footprint.

## Features

- 🤗 Seamless integration with Hugging Face Transformers
- 🚀 PEFT/LoRA support for memory-efficient training
- 🤖 Compatible with both causal and seq2seq models
- 🔥 Optimized for various hardware configurations


## Quick Tour

### Installation

```bash
pip install cycleformers
```

### Training

The `CycleTrainer` class is an extension but significant redesign of the 🤗 Transformers trainer, designed to abstract away the specifics of training while remaining configurable. Both Seq2Seq and Causal architectures are supported, each able to train via PEFT adapter swapping for memory efficient configurations. Check the [docs] for [usage] details and [examples].

To train using two identical models the following sample code can be used along with two datasets:

```python
from cycleformers import CycleTrainer, CycleTrainingArguments

model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

args = CycleTrainingArguments(output_dir="gpt2-cct")
trainer = CycleTrainer(
    args, 
    models = model
    tokenizers = tokenizer
    train_dataset_A = dataset_A,
    train_dataset_B = dataset_B
)
trainer.train()
```

Any two models (🚧 currently both seq2seq or both causal) can be combined together for completely customisable training:

```python
model_A = AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto")
model_B = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", device_map="auto")
tokenizer_A = AutoTokenizer.from_pretrained("gpt2")
tokenizer_B = AutoTokenizer.from_pretrained("google/flan-t5-small")

trainer = CycleTrainer(
    args, 
    models = {
        "A": model_A,
        "B": model_B
    }
    tokenizers = {
        "A": tokenizer_A,
        "B": tokenizer_B
    }
    train_dataset_A = dataset_A,
    train_dataset_B = dataset_B
)
```

### Multi-Adapter Cycle-Consistency Training (MACCT)

The `CycleTrainer` class is also setup to accept a single base model and train two PEFT adapters ontop of it, switching between them to emulate the two model setup. This allows for the training of `7.5x larger models` for the same memory footprint:

```python
peft_config = PeftConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    inference_mode=False,
    bias="none"
)

args = CycleTrainingArguments(output_dir="gpt2-macct")
trainer = CycleTrainer(
    args, 
    model = model,
    tokenizer = tokenizer,
    peft_configs = peft_config # Or same A, B dict
)
```



## Citing

If you use Cycleformers in your research, please cite:

```bibtex
add once zenodo/paper citation is available
```