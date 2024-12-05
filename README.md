# Cycleformers

<div align="center">

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Transformers](https://img.shields.io/badge/🤗_transformers-4.46.1-yellow.svg)](https://github.com/huggingface/transformers)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-EE4C2C.svg)](https://pytorch.org/)

<!-- [![Build Status](https://github.com/wrmthorne/cycleformers/workflows/CI-Pipeline/badge.svg)](https://github.com/wrmthorne/cycleformers/actions) -->

</div>

Cycleformers exposes a high-level but finely configurable API for transformer-based cycle-consistent architectures. The primary objective of the library is to provide a very simple framework to start using that works, out-of-the-box on as wide a range of hardware configurations as possible. This will enable quick iteration in training and research. The priority of flexibility is not limited to hardware - we aim to offer an interface that enforces no restrictions on the data, models (or number thereof), and enable immediate access to the latest versions of the Huggingface ecosystem. The fragmentation of training scripts for cycle-consistency across different domains, backend packages, and hardware selections has required frequent reimplementaion of the same idea to resolve implementation level details when the goal is to investigate the applications of the paradigm.

`NOTE:` All existing APIs are subject to change without notice.

## Features

- 🚀 High-performance transformer implementations
- 🔄 Cycle-consistent architecture
- 📊 A pure extension of the Huggingface ecosystem
- 🛠️ Flexible and extensible design
- 📝 Comprehensive documentation and examples

## In Progress (Ordered by Priority)

- ✅ Causal-to-Causal cycle implementation
- ✅ Adding peft into the main trainer rather than using a more complicated trainer thats too bespoke
- 🚧 Saving and loading from checkpoints
- 🚧 arbitrary model-to-model training (mixed architectures, model families, etc.)
- 🚧 Improved evaluation metrics

All throughout, api improvements, QoL improvements, and documentation improvements. Points 1-3 are required for a pre-release.

Backlog:
- Specific configuration for each model through cli args specified in dataclasses
Allow custom naming of each cycle through keys of dicts for models, tokenizers, etc.
- Rework of the current flow control classes as global state is currently hacked to only monitor model_A
- Tokenizer-mapping for arbitrary model-to-model training ([github](https://github.com/explosion/tokenizations/blob/master/note/blog_post.md), [ArXiv](https://arxiv.org/html/2411.00593v1))
- Investigation of potential tokenization issues e.g. not generating EOS tokens in training examples so will never learn to stop, separator token/whitespace for causal cycles
- Handling very different batch sizes for each model in the cycle. May happen that the batch size output from one model is too big to pass in one generate/train call. Need to batch the output of one model to be used as input to the next.

## CycleTrainer

```python
from cycleformers import CycleTrainer, CycleTrainingArguments

# Arbitrary combinations of generative models
model_A = AutoModelForCausalLM.from_pretrained("gpt2")
model_B = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

...

args = CycleTrainingArguments()
trainer = CycleTrainer(
    args, 
    models = {
        "A": model_A,
        "B": model_B
    }
    tokenizers = {
        "A": tokenizer_A,
        "B": tokenizer_B
    },
    train_dataset_A = train_dataset_A,
    train_dataset_B = train_dataset_B
)
trainer.train()
```

## Multi-Adapter Cycle-Consistency Training (MACCT)

```python

```
A tested script using the intended API is given at [examples/scripts/macct.py](https://github.com/wrmthorne/cycleformers/tree/main/examples/scripts/macct.py).


## Conceptual Clarifications

WILL PROBABLY CHANGE

There are some terms used throughout that may differ based on opinion. These are to explain how I have interpreted them:
- `global_step` one call of the dataloaders i.e. (batch_a, batch_b) in zip(dataloader_a, dataloader_b).
- `epoch` one full pass through the zipped dataloaders i.e. min(len(dataloader_a), len(dataloader_b)) calls of the dataloaders.

## Current Limitations

While developing from this early version, there are some constraints imposed on the user while the library matures. The most notable is that Multi-Adapter Cycle-Consistency Training is the only supported mode of training as it was *shockingly* the easiest to implement and validate in a way that's almost (note upcoming caveats) compatible with the standard huggingface trainer.

### Interleaving

- Gradient accumulation steps are doubled and datasets are cycled in sequence to ensure gradients aren't zeroed before an adapter was able to run an optimiser step.
- [FINISH BEFORE PRERELEASE PUBLICATION]

## Citing

If you use Cycleformers in your research, please cite:

```bibtex
```