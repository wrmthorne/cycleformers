from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_false", help="Run slow tests")
    parser.addoption("--all", action="store_false", help="Run all (possible) tests")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--all"):
        skip_slow = pytest.mark.skip(reason="specify --slow or --all to run")
        for item in items:
            if "slow" in item.keywords and not config.getoption("--slow"):
                item.add_marker(skip_slow)
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="No GPU available")
        for item in items:
            if "requires_gpu" in item.keywords:
                item.add_marker(skip_gpu)


# TODO: replace with tiny-random-model
# TODO: add config tests for other architectures
@dataclass
class Seq2SeqModelTestConfig:
    model_name_or_path: str = "google/t5-efficient-tiny"


@dataclass
class CausalModelTestConfig:
    model_name_or_path: str = "trl-internal-testing/tiny-LlamaForCausalLM-3.1"  # TODO: Make own tiny models


@pytest.fixture(name="seq2seq_config")
def fixture_seq2seq_config() -> Seq2SeqModelTestConfig:
    return Seq2SeqModelTestConfig()


@pytest.fixture(name="causal_config")
def fixture_causal_config() -> CausalModelTestConfig:
    return CausalModelTestConfig()


@pytest.fixture(name="text")
def fixture_text() -> str:
    return "This is a test input"


@pytest.fixture(name="seq2seq_model")
def fixture_seq2seq_model(seq2seq_config):
    return AutoModelForSeq2SeqLM.from_pretrained(seq2seq_config.model_name_or_path)


@pytest.fixture(name="causal_model")
def fixture_causal_model(causal_config):
    return AutoModelForCausalLM.from_pretrained(causal_config.model_name_or_path)


@pytest.fixture(name="peft_causal_model")
def fixture_peft_causal_model(causal_model):
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
    )
    model = get_peft_model(causal_model, lora_config, adapter_name="A")
    model.add_adapter("B", lora_config)
    return model


@pytest.fixture(name="seq2seq_tokenizer")
def fixture_seq2seq_tokenizer(seq2seq_config):
    return AutoTokenizer.from_pretrained(seq2seq_config.model_name_or_path)


@pytest.fixture(name="causal_tokenizer")
def fixture_causal_tokenizer(causal_config):
    tokenizer = AutoTokenizer.from_pretrained(causal_config.model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return tokenizer


@pytest.fixture(name="optimizer")
def fixture_optimizer():
    return AdamW


@pytest.fixture(name="seq2seq_inputs")
def fixture_seq2seq_inputs(text, seq2seq_tokenizer):
    return seq2seq_tokenizer(text, return_tensors="pt")


@pytest.fixture(name="text_dataset")
def fixture_text_dataset():
    text = ["This is a test input", "This is another test input", "This is a third and final test input"]
    labels = [
        "This is an acknowledgement",
        "This is another acknowledgement",
        "This is a third and final acknowledgement",
    ]
    return Dataset.from_dict({"text": text, "labels": labels})


@pytest.fixture(name="tokenized_dataset")
def fixture_tokenized_dataset(causal_tokenizer, text_dataset):
    dataset = text_dataset.map(
        lambda x: {
            **causal_tokenizer(x["text"], return_tensors="pt"),
            "labels": causal_tokenizer(x["labels"], return_tensors="pt")["input_ids"],
        }
    ).remove_columns(["text"])
    return dataset
