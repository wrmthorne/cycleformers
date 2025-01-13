# Cycleformers

<div align="center">

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/cycleformers)](https://pypi.org/project/cycleformers/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Coverage](.github/badges/coverage.svg)](https://codecov.io/gh/wrmthorne/cycleformers)
[![Build Status](.github/badges/build.svg)](https://github.com/wrmthorne/cycleformers/actions/workflows)
[![Documentation Status](https://readthedocs.org/projects/cycleformers/badge/?version=latest)](https://cycleformers.readthedocs.io/en/latest/?badge=latest)

</div>

A Python library for efficient cycle-consistency training of transformer models. Cycleformers simplifies iterative back-translation with support for both causal and seq2seq architectures. We also implement Multi-Adapter Cycle-Consistency Training (MACCT), enabling training of LoRA adapters on a frozen base model for `7.5x` larger model capacity for the same memory footprint.

## 🌟 Features

- 🤗 Seamless integration with Hugging Face Transformers
- 🚀 PEFT/LoRA support for memory-efficient training
- 🤖 Compatible with both causal and seq2seq models
- 🔥 Optimized for various hardware configurations
- 📚 Comprehensive documentation and examples
- 🧪 High test coverage and reliability

## 🚀 Quick Start

### Installation

```bash
pip install cycleformers
```

### Basic Usage

The `CycleTrainer` extends the 🤗 Transformers trainer with cycle-consistency capabilities. Here's a minimal example:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from cycleformers import CycleTrainer, CycleTrainingArguments

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Configure training
args = CycleTrainingArguments(output_dir="gpt2-cct")
trainer = CycleTrainer(
    args=args, 
    models=model,  # Will be duplicated internally
    tokenizers=tokenizer,
    train_dataset_A=dataset_A,
    train_dataset_B=dataset_B
)

# Start training
trainer.train()
```

### Advanced: Different Models

Use two different models (must be same architecture type):

```python
model_A = AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto")
model_B = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", device_map="auto")
tokenizer_A = AutoTokenizer.from_pretrained("gpt2")
tokenizer_B = AutoTokenizer.from_pretrained("facebook/opt-125m")

trainer = CycleTrainer(
    args=args, 
    models={"A": model_A, "B": model_B},
    tokenizers={"A": tokenizer_A, "B": tokenizer_B},
    train_dataset_A=dataset_A,
    train_dataset_B=dataset_B
)
```

### Multi-Adapter Cycle-Consistency Training (MACCT)

Train a single larger model with two PEFT adapters, using ~7.5x less memory:

```python
from peft import LoraConfig

# Configure LoRA adapters
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    bias="none"
)

args = CycleTrainingArguments(
    output_dir="gpt2-macct",
    use_macct=True  # Enable adapter-based training
)

trainer = CycleTrainer(
    args=args, 
    models=model,
    tokenizers=tokenizer,
    peft_configs=peft_config  # Will be used for both adapters
)
```

## 📚 Documentation

- [Full Documentation](https://cycleformers.readthedocs.io/)
- [API Reference](https://cycleformers.readthedocs.io/en/latest/api_reference/)
- [Examples](https://wrmthorne.github.io/cycleformers/examples/)
- [Contributing Guide](https://cycleformers.readthedocs.io/en/latest/contributing/)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📝 Citation

If you use Cycleformers in your research, please cite:

```bibtex
add once zenodo/paper citation is available
```

## 📄 License

This project is licensed under the CC BY 4.0 License - see the [LICENSE](LICENSE) file for details.