import sys
from dataclasses import dataclass

import pytest
import yaml

from cycleformers.command.cli_utils import TASKS, CFArgumentParser


@dataclass
class DummyArgs:
    model_name: str
    learning_rate: float


@dataclass
class DummyArgsAB:
    A_model: str
    B_model: str
    A_learning_rate: float
    B_learning_rate: float
    shared_param: str


class TestCFArgumentParser:
    def test_init_with_task(self):
        parser = CFArgumentParser(task="train")
        assert parser.task == "train"

    def test_init_without_task(self):
        parser = CFArgumentParser()
        assert parser.task is None

    def test_invalid_task_raises_error(self, monkeypatch):
        parser = CFArgumentParser()
        monkeypatch.setattr(sys, "argv", ["invalid_task"])
        with pytest.raises(ValueError, match="Task must be one of"):
            parser.parse_args_and_config()

    def test_parse_simple_args(self, monkeypatch, tmp_path):
        parser = CFArgumentParser([DummyArgs])
        args = ["train", "--model_name", "bert", "--learning_rate", "0.001"]
        monkeypatch.setattr(sys, "argv", ["script.py"] + args)

        (parsed_args,) = parser.parse_args_and_config()
        assert isinstance(parsed_args, DummyArgs)
        assert parsed_args.model_name == "bert"
        assert parsed_args.learning_rate == 0.001

    def test_parse_yaml_config(self, tmp_path):
        config = {
            "A": {"model": "bert-base", "learning_rate": 0.001},
            "B": {"model": "gpt2", "learning_rate": 0.0001},
            "shared_param": "value",
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        parser = CFArgumentParser([DummyArgsAB])
        file_args = parser._parse_yaml_config(str(config_path))

        # Convert file_args list to dict for easier assertion
        args_dict = dict(zip(file_args[::2], file_args[1::2]))

        assert args_dict["--A_model"] == "bert-base"
        assert args_dict["--B_model"] == "gpt2"
        assert args_dict["--A_learning_rate"] == "0.001"
        assert args_dict["--B_learning_rate"] == "0.0001"
        assert args_dict["--shared_param"] == "value"

    def test_yaml_with_cli_override(self, monkeypatch, tmp_path):
        config = {"A": {"model": "bert-base"}, "B": {"model": "gpt2"}}

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        parser = CFArgumentParser([DummyArgsAB])
        args = [
            "train",
            str(config_path),
            "--A_learning_rate",
            "0.001",
            "--B_learning_rate",
            "0.0001",
            "--shared_param",
            "override",
        ]
        monkeypatch.setattr(sys, "argv", ["script.py"] + args)

        (parsed_args,) = parser.parse_args_and_config()
        assert parsed_args.A_model == "bert-base"
        assert parsed_args.B_model == "gpt2"
        assert parsed_args.A_learning_rate == 0.001
        assert parsed_args.B_learning_rate == 0.0001
        assert parsed_args.shared_param == "override"

    def test_duplicate_args_in_yaml(self, tmp_path):
        config = {
            "A": {"param": "value1"},
            "A_param": "value2",  # This creates a duplicate A_param
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        parser = CFArgumentParser()
        with pytest.raises(ValueError, match="Duplicate argument"):
            parser._parse_yaml_config(str(config_path))
