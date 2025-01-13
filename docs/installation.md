# Installation

## Quick Start

Cycleformers is available on PyPi and can be installed with pip:

```bash
pip install cycleformers
```

## System Requirements

- **Python Version**: 3.11 or higher
- **Operating System**: Linux (Ubuntu 20.04 or newer recommended)
- **GPU**: Optional but recommended for training larger models
- **RAM**: Minimum 16GB recommended (varies based on model size)

!!! note "Platform Support"
    The module has been extensively tested on Linux. Windows and MacOS are not officially supported at this time.

## Development Installation

For contributors and developers who want to modify the codebase, we provide a development setup:

1. Clone the repository:
```bash
git clone https://github.com/wrmthorne/cycleformers.git
cd cycleformers
```

2. Initialise the development environment:
```bash
make init
```

This command will:
- Create a Python virtual environment
- Install all dependencies via Poetry
- Set up pre-commit hooks for code quality
- Initialize development configurations

## Troubleshooting

If you encounter any installation issues:

1. Ensure you have the latest pip version:
```bash
python -m pip install --upgrade pip
```

2. Check your Python version:
```bash
python --version
```

3. For development setup issues, ensure you have Make and Poetry installed:
```bash
# Install Poetry if needed
curl -sSL https://install.python-poetry.org | python3 -
```

For additional help, please [open an issue](https://github.com/wrmthorne/cycleformers/issues) on our GitHub repository.