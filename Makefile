# https://medium.com/@dkraczkowski/crafting-a-ci-pipeline-my-experience-with-github-actions-python-aws-c67f428adee8
-include .env
SOURCE_DIR = src
TEST_DIR = tests
PROJECT_DIRS = $(SOURCE_DIR) $(TEST_DIR)
PWD := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))
PROJECT_NAME ?= CycleFormers
PROJECT_VERSION ?= v$(shell poetry version -s)
PYTHON_VERSION ?= 3.10
.DEFAULT_GOAL := all

.PHONY: init-env init check-toml lint-src format lint audit test all clean info

init-env:
	@echo "Creating .env file..."
	@touch .env
	@echo "PROJECT_NAME=${PROJECT_NAME}" >> .env
	@echo "PYTHON_VERSION=${PYTHON_VERSION}" >> .env

init: init-env
	@echo "Installing dependencies..."
	poetry install

-check-toml:
	poetry check

-lint-src:
	poetry run ruff check --fix $(SOURCE_DIR)
	poetry run mypy --install-types --show-error-codes --non-interactive $(SOURCE_DIR)

format: -check-toml -reformat-src

lint: -lint-src

audit:
	poetry run bandit -r $(SOURCE_DIR) -x $(TEST_DIR)

test:
	poetry run pytest $(TEST_DIR)

all: format lint audit test

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