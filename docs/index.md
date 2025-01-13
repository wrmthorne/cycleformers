# Cycleformers

<div align="center">

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/cycleformers)](https://pypi.org/project/cycleformers/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Coverage](.github/badges/coverage.svg)](https://codecov.io/gh/wrmthorne/cycleformers)
[![Build Status](.github/badges/build.svg)](https://github.com/wrmthorne/cycleformers/actions/workflows)

</div>

A Python library for efficient cycle-consistency training of transformer models. Cycleformers simplifies iterative back-translation with support for both causal and seq2seq architectures. We also implement Multi-Adapter Cycle-Consistency Training (MACCT), enabling training of LoRA adapters on a frozen base model for `7.5x` larger model capacity for the same memory footprint.

## Features

- 🤗 Seamless integration with Hugging Face Transformers
- 🚀 PEFT/LoRA support for memory-efficient training
- 🤖 Compatible with both causal and seq2seq models
- 🔥 Optimized for various hardware configurations

## Documentation

- [Conceptual Reference](conceptual_reference/task_processors.md)
- [API Reference](api_reference/task_processors.md)
- [Examples](examples/index.md)
