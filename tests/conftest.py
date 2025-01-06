import random
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from cycleformers.import_utils import is_peft_available

from .testing_utils.model_registry import CapabilityExpression, ModelCapability, ModelRegistry, ModelSpec


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


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Automatically set seeds before each test."""
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)


@pytest.fixture(scope="session")
def model_registry() -> ModelRegistry:
    """Record of all models used for testing."""
    return ModelRegistry(Path(pytest.config.getoption("--model-registry")))


@lru_cache(maxsize=None)
def load_model_and_tokenizer(model_spec: ModelSpec) -> PreTrainedModel:
    """Cache model loading to speed up tests."""
    tokenizer = AutoTokenizer.from_pretrained(model_spec.repo_id)

    if ModelCapability.SEQ2SEQ in model_spec.capabilities:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_spec.repo_id)
    elif ModelCapability.CAUSAL in model_spec.capabilities:
        model = AutoModelForCausalLM.from_pretrained(model_spec.repo_id)
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    else:
        raise ValueError(f"No model found for {model_spec.repo_id} with capabilities {model_spec.capabilities}")

    return model, tokenizer


def get_specific_model(
    model_registry: ModelRegistry,
    capabilities: ModelCapability | CapabilityExpression | None = None,
    model_names: list[str] | str | None = None,
) -> ModelSpec:
    """Returns a pytest fixture that parametrizes over all models matching the given capabilities and names."""
    if pytest.config.getoption("--all"):
        params = model_registry.get_matching_model(capabilities, model_names)
    elif pytest.config.getoption("--slow"):
        params = model_registry.get_random_model()
    else:
        pytest.skip("Need --slow or --all option to run.")

    @pytest.fixture(params=params)
    def _models_and_tokenizers(request) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        model_spec = request.param
        if not isinstance(model_spec, ModelSpec):
            pytest.skip(f"No models found with capabilities={capabilities} and names={model_names}")
        return load_model_and_tokenizer(model_spec)

    return _models_and_tokenizers


def get_specific_peft_model(
    model_registry: ModelRegistry,
    capabilities: ModelCapability | CapabilityExpression | None = None,
    model_names: list[str] | str | None = None,
    peft_config: LoraConfig | None = None,
) -> ModelSpec:
    """Returns a pytest fixture that parametrizes over all PEFT-adapted models matching the given capabilities and names."""
    if not is_peft_available():
        pytest.skip("PEFT is not installed")

    base_model_fixture = get_specific_model(model_registry, capabilities, model_names)
    peft_config = peft_config or LoraConfig(r=8, lora_alpha=16)

    @pytest.fixture(params=base_model_fixture.params)
    def _peft_models(request) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        model, tokenizer = base_model_fixture(request)
        if isinstance(model, AutoModelForCausalLM):
            peft_config.task_type = "CAUSAL_LM"
        elif isinstance(model, AutoModelForSeq2SeqLM):
            peft_config.task_type = "SEQ_2_SEQ_LM"
        return get_peft_model(model, peft_config), tokenizer

    return _peft_models


# ===== Common model fixtures ===== #
any_model, any_tokenizer = get_specific_model()  # All models in the registry
seq2seq_model, seq2seq_tokenizer = get_specific_model(capabilities=ModelCapability.SEQ2SEQ)
causal_model, causal_tokenizer = get_specific_model(capabilities=ModelCapability.CAUSAL)

if is_peft_available():
    any_peft_model, any_peft_tokenizer = get_specific_peft_model()
    seq2seq_peft_model, seq2seq_peft_tokenizer = get_specific_peft_model(capabilities=ModelCapability.SEQ2SEQ)
    causal_peft_model, causal_peft_tokenizer = get_specific_peft_model(capabilities=ModelCapability.CAUSAL)
