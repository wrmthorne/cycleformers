# CycleFormers

<div align="center">

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
<!-- [![Build Status](https://github.com/wrmthorne/CycleFormers/workflows/CI-Pipeline/badge.svg)](https://github.com/wrmthorne/CycleFormers/actions) -->

</div>

CycleFormers exposes a high-level but finely configurable API for transformer-based cycle-consistent architectures. The primary objective of the library is to provide a very simple framework to start using that works, out-of-the-box on as wide a range of hardware configurations as possible. This will enable quick iteration in training and research. The priority of flexibility is not limited to hardware - we aim to offer an interface that enforces no restrictions on the data, models (or number thereof), and enable immediate access to the latest versions of the Huggingface ecosystem. The fragmentation of training scripts for cycle-consistency across different domains, backend packages, and hardware selections has required frequent reimplementaion of the same idea to resolve implementation level details when the goal is to investigate the applications of the paradigm.

## Features

- 🚀 High-performance transformer implementations
- 🔄 Cycle-consistent architecture
- 📊 A pure extension of the Huggingface ecosystem
- 🛠️ Flexible and extensible design
<!-- - 📝 Comprehensive documentation and examples -->

## Installation

```bash
pip install CycleFormers
```

## Citing

If you use CycleFormers in your research, please cite:

```bibtex
```