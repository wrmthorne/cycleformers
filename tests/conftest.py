import random
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Generator, Optional, Tuple, Union

import numpy as np
import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from cycleformers.import_utils import is_peft_available

from .testing_utils.model_registry import (
    CapabilityExpression,
    ModelCapability,
    ModelRegistry,
    ModelSpec,
)


if is_peft_available():
    from peft import LoraConfig, get_peft_model

DEFAULT_MODEL_REGISTRY_PATH = Path(__file__).parent / "models_to_test.yaml"


def pytest_addoption(parser):
    parser.addoption(
        "--slow", action="store_true", help="Run slow tests. Randomly selects one valid model for each test."
    )
    parser.addoption("--meta", action="store_true", help="Run tests of testing harness components.")
    parser.addoption(
        "--all", action="store_true", help="Run all tests possible on hardware. Runs all valid models for each test."
    )
    parser.addoption("--no-gpu", action="store_false", help="Skip tests that require a GPU.")
    parser.addoption(
        "--model-registry", action="store", default=DEFAULT_MODEL_REGISTRY_PATH, help="Path to model registry"
    )


def pytest_collection_modifyitems(config, items):
    run_all = config.getoption("--all")
    run_slow = config.getoption("--slow") or run_all
    run_meta = config.getoption("--meta") or run_all
    run_gpu = not config.getoption("--no-gpu") and torch.cuda.is_available()

    skip_slow = pytest.mark.skip(reason="Need --slow or --all option to run")
    skip_meta = pytest.mark.skip(reason="Need --meta or --all option to run")
    skip_gpu = pytest.mark.skip(reason="Need GPU to run and --no-gpu not set")

    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        if "meta" in item.keywords and not run_meta:
            item.add_marker(skip_meta)
        if "gpu" in item.keywords and not run_gpu:
            item.add_marker(skip_gpu)


@pytest.fixture
def temp_dir():
    """Provide temporary directory that's cleaned up after test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
        shutil.rmtree(tmpdir)


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Automatically set seeds before each test."""
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)


@pytest.fixture(scope="session")
def model_registry(request) -> ModelRegistry:
    """Record of all models used for testing."""
    return ModelRegistry(Path(request.config.getoption("--model-registry")))


@lru_cache(maxsize=None)
def load_model_and_tokenizer(model_spec: ModelSpec) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Cache model loading to speed up tests."""
    tokenizer = AutoTokenizer.from_pretrained(model_spec.repo_id)

    if ModelCapability.SEQ2SEQ_LM in model_spec.capabilities:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_spec.repo_id)
    elif ModelCapability.CAUSAL_LM in model_spec.capabilities:
        model = AutoModelForCausalLM.from_pretrained(model_spec.repo_id)
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    else:
        raise ValueError(f"No model found for {model_spec.repo_id} with capabilities {model_spec.capabilities}")

    return model, tokenizer


def create_model_fixture(
    capabilities: Optional[Union[ModelCapability, CapabilityExpression]] = None,
    model_names: Optional[Union[list[str], str]] = None,
    is_peft: bool = False,
    peft_config: Optional[LoraConfig] = None,
) -> pytest.fixture:
    """Factory function to create model fixtures with specified capabilities."""

    @pytest.fixture
    def model_fixture(request, model_registry) -> Generator[Tuple[PreTrainedModel, PreTrainedTokenizer], None, None]:
        if not (request.config.getoption("--all") or request.config.getoption("--slow")):
            pytest.skip("Need --slow or --all option to run")

        if capabilities is not None or model_names is not None:
            models = model_registry.get_matching_models(capabilities, model_names)
        else:
            models = model_registry.get_matching_models()

        if not models:
            pytest.skip(f"No models found with capabilities={capabilities} and names={model_names}")

        # If --slow but not --all, randomly select one model
        if not request.config.getoption("--all"):
            models = [random.choice(models)]

        for model_spec in models:
            model, tokenizer = load_model_and_tokenizer(model_spec)

            if is_peft:
                if not is_peft_available():
                    pytest.skip("PEFT is not installed")

                config = peft_config or LoraConfig(r=8, lora_alpha=16)
                config.task_type = "CAUSAL_LM" if isinstance(model, AutoModelForCausalLM) else "SEQ_2_SEQ_LM"
                model = get_peft_model(model, config)

            yield model, tokenizer

    return model_fixture


# Common model fixtures
any_model_and_tokenizer = create_model_fixture()
seq2seq_model_and_tokenizer = create_model_fixture(capabilities=ModelCapability.SEQ2SEQ_LM)
causal_model_and_tokenizer = create_model_fixture(capabilities=ModelCapability.CAUSAL_LM)
random_model_and_tokenizer = create_model_fixture()

# PEFT model fixtures
if is_peft_available():
    any_peft_model_and_tokenizer = create_model_fixture(is_peft=True)
    seq2seq_peft_model_and_tokenizer = create_model_fixture(capabilities=ModelCapability.SEQ2SEQ_LM, is_peft=True)
    causal_peft_model_and_tokenizer = create_model_fixture(capabilities=ModelCapability.CAUSAL_LM, is_peft=True)
    random_peft_model_and_tokenizer = create_model_fixture(is_peft=True)
