
# Contributing Guide

We are interested in external contributions to Cycleformers, particularly with those experienced in automated testing of language models, and with multi-device training.

Please contribute code via pull requests. For small bug fixes or improvements, please feel free to submit a PR directly to the main branch. For larger bugs or features, please open an issue for discussion first. If you would like to request a feature, please also open an issue. For any PRs, please ensure that you have added tests and documentation where appropriate.

All new features should be developed on a new branch and use [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) prefixes. The following are recommended:

- `fix`: for bug fixes
- `feat`: for new features
- `build:` for changes that affect the build system or external dependencies
- `refactor:` for code refactoring
- `ci:` for changes to CI configuration files and scripts
- `docs:` for documentation changes
- `perf:` for performance improvements
- `test:` for adding tests
- `chore:` for other changes that don't fit into the other categories

Although this rule is not strictly enforced for commits, pull requests must be titled with a valid conventional commit prefix or the merge pipeline will fail. Pull requests should be made to main and will not be released immediately. Changes will be accumulated on main until a release is decided. The release workflow will handle versioning, publishing, changelog generation, etc. and is triggered by a release workflow dispatch event, selecting `main` as the branch.

## Development Setup

We use Poetry to manage dependencies on Python 3.11. To install Poetry, follow the documentation here: https://python-poetry.org/docs/master/#installing-with-the-official-installer

We recommend the following command to setup the build environment for the module:

```bash
make init
```

Many other useful commands are available in the Makefile which will automate formatting, linting, testing and building the documentation:

- `make format` automatically fixes code style issues and standardizes formatting across the codebase.
- `make lint` checks for code quality issues, potential bugs, and style violations.
- `make audit` performs security vulnerability scanning to identify potential security risks.
- `make test` executes the automated test suite to verify code functionality.
- `make build-docs` generates the HTML documentation from source files based on the configuration in `mkdocs.yml`.
- `make all` runs all of the above commands in sequence.
- `make clean` removes build and distribution artifacts.
- `make info` displays information about the project, including the project name, version, and Python version.


## Project Procedures

 