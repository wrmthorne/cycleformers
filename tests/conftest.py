from dataclasses import dataclass

import pytest
from peft import LoraConfig
from torch.optim import AdamW
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM

from cycleformers.models.peft_models import CycleAdapterConfig, PeftCycleModelForSeq2SeqLM


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", help="Run slow tests")
    parser.addoption("--meta", action="store_true", help="Run meta tests")
    parser.addoption(
        "--all", action="store_false", help="Run all (possible) tests"
    )  # TODO: Change back to store_true when ready


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--all"):
        skip_slow = pytest.mark.skip(reason="specify --slow or --all to run")
        skip_meta = pytest.mark.skip(reason="specify --meta or --all to run")
        for item in items:
            if "slow" in item.keywords and not config.getoption("--slow"):
                item.add_marker(skip_slow)
            if "meta" in item.keywords and not config.getoption("--meta"):
                item.add_marker(skip_meta)


# TODO: replace with tiny-random-model
# TODO: add config tests for other architectures
@dataclass
class Seq2SeqModelTestConfig:
    base_model_name: str = "google/t5-efficient-tiny"


@dataclass
class CausalModelTestConfig:
    base_model_name: str = "trl-internal-testing/tiny-LlamaForCausalLM-3.1" # TODO: Make own tiny models


@pytest.fixture(name="lora_config")
def fixture_lora_config(request):
    task_type = getattr(request, "param", "SEQ_2_SEQ_LM")
    return LoraConfig(
        task_type=task_type,
        r=8,
        lora_alpha=32,
        target_modules=("q", "v"),
    )


@pytest.fixture(name="seq2seq_config")
def fixture_seq2seq_config() -> Seq2SeqModelTestConfig:
    return Seq2SeqModelTestConfig()


@pytest.fixture(name="causal_config")
def fixture_causal_config() -> CausalModelTestConfig:
    return CausalModelTestConfig()


@pytest.fixture(name="text")
def fixture_text() -> str:
    return "This is a test input"


@pytest.fixture(name="seq2seq_base_model")
def fixture_seq2seq_base_model(seq2seq_config):
    return AutoModelForSeq2SeqLM.from_pretrained(seq2seq_config.base_model_name)


@pytest.fixture(name="causal_base_model")
def fixture_causal_base_model(causal_config):
    return AutoModelForCausalLM.from_pretrained(causal_config.base_model_name)


@pytest.fixture(name="seq2seq_tokenizer")
def fixture_seq2seq_tokenizer(seq2seq_config):
    return AutoTokenizer.from_pretrained(seq2seq_config.base_model_name)


@pytest.fixture(name="causal_tokenizer")
def fixture_causal_tokenizer(causal_config):
    tokenizer = AutoTokenizer.from_pretrained(causal_config.base_model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return tokenizer


@pytest.fixture(name="seq2seq_model")
def fixture_seq2seq_model(seq2seq_base_model, seq2seq_tokenizer, lora_config):
    adapter_a_config = CycleAdapterConfig(adapter_name="adapter_a", peft_config=lora_config)
    adapter_b_config = CycleAdapterConfig(adapter_name="adapter_b", peft_config=lora_config)

    base_model = PeftCycleModelForSeq2SeqLM(
        model=seq2seq_base_model, tokenizer=seq2seq_tokenizer, adapter_configs=[adapter_a_config, adapter_b_config]
    )
    return base_model


@pytest.fixture(name="seq2seq_inputs")
def fixture_seq2seq_inputs(text, seq2seq_tokenizer):
    return seq2seq_tokenizer(text, return_tensors="pt")


@pytest.fixture(name="optimizer")
def fixture_optimizer():
    return AdamW
