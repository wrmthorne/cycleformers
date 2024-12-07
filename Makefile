# https://medium.com/@dkraczkowski/crafting-a-ci-pipeline-my-experience-with-github-actions-python-aws-c67f428adee8
-include .env
SOURCE_DIR = src
TEST_DIR = tests
EXAMPLE_DIR = examples
PROJECT_DIRS = $(SOURCE_DIR) $(TEST_DIR) $(EXAMPLE_DIR)
PWD := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))
PROJECT_NAME ?= Cycleformers
PROJECT_VERSION ?= v$(shell poetry version -s)
PYTHON_VERSION ?= 3.11
.DEFAULT_GOAL := all

.PHONY: init-env init check-toml lint-src format lint audit test all clean info build-docs

init-env:
	@if [ ! -f .env ]; then \
		echo "Creating .env file..."; \
		echo "PROJECT_NAME=${PROJECT_NAME}" > .env; \
		echo "PYTHON_VERSION=${PYTHON_VERSION}" >> .env; \
		echo "export PYTHONPATH=${SOURCE_DIR}" >> .env; \
	else \
		echo "using existing .env file..."; \
	fi

init: init-env
	@echo "Installing dependencies..."
	poetry install

-check-toml:
	poetry check

-reformat-src:
	poetry run ruff format $(PROJECT_DIRS)
	poetry run ruff check --select I --fix $(PROJECT_DIRS)

-lint-src:
	poetry run ruff check --fix $(SOURCE_DIR)
	poetry run mypy --install-types --show-error-codes --non-interactive $(SOURCE_DIR)

format: -check-toml -reformat-src

lint: -lint-src

audit:
	poetry run bandit -r $(SOURCE_DIR) -x $(TEST_DIR)

test:
	poetry run pytest $(TEST_DIR)

build-docs:
	poetry run mkdocs build

all: format lint audit test build-docs

clean:
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -exec rm -rf {} +
	find . -type d -name '.mypy_cache' -exec rm -rf {} +

info:
	@echo "Project name: ${PROJECT_NAME}"
	@echo "Project version: ${PROJECT_VERSION}"
	@echo "Python version: ${PYTHON_VERSION}"