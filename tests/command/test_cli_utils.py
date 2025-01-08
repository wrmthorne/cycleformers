import sys
from dataclasses import dataclass
from typing import Tuple

import pytest
import yaml

from cycleformers.command.cli_utils import VALID_TASKS, CfArgumentParser


@dataclass
class DummyArgs:
    model_name: str
    learning_rate: float = 0.001


@dataclass
class DummyArgsAB:
    A_model: str
    B_model: str
    A_learning_rate: float = 0.001
    B_learning_rate: float = 0.0001
    shared_param: str = "default"


@dataclass
class DummyArgsC:
    C_model: str
    C_learning_rate: float = 0.0005


class TestCFArgumentParser:
    def test_init_with_task(self):
        parser = CfArgumentParser(task="train")
        assert parser.task == "train"

    def test_init_without_task(self):
        parser = CfArgumentParser()
        assert parser.task is None

    def test_init_with_multiple_dataclasses(self):
        parser = CfArgumentParser([DummyArgs, DummyArgsAB])
        assert len(parser.dataclass_types) == 2
        assert parser.dataclass_types[0] == DummyArgs
        assert parser.dataclass_types[1] == DummyArgsAB

    def test_init_with_single_dataclass_not_in_list(self):
        parser = CfArgumentParser(DummyArgs)
        assert len(parser.dataclass_types) == 1
        assert parser.dataclass_types[0] == DummyArgs

    def test_init_with_no_dataclasses(self):
        parser = CfArgumentParser()
        assert len(parser.dataclass_types) == 0

    def test_invalid_task_raises_error(self, monkeypatch):
        parser = CfArgumentParser()
        monkeypatch.setattr(sys, "argv", ["invalid_task"])
        with pytest.raises(ValueError, match="Task must be one of"):
            parser.parse_args_and_config()

    def test_no_task_provided_raises_error(self, monkeypatch):
        parser = CfArgumentParser()
        monkeypatch.setattr(sys, "argv", ["script.py"])
        with pytest.raises(ValueError, match="No task provided"):
            parser.parse_args_and_config()

    def test_task_mismatch_raises_error(self, monkeypatch):
        parser = CfArgumentParser(task="train")
        monkeypatch.setattr(sys, "argv", ["script.py", "train"])
        with pytest.raises(ValueError, match="Task already set"):
            parser.parse_args_and_config()

    def test_parse_simple_args(self, monkeypatch):
        parser = CfArgumentParser([DummyArgs])
        args = ["train", "--model_name", "bert", "--learning_rate", "0.001"]
        monkeypatch.setattr(sys, "argv", ["script.py"] + args)

        (parsed_args,) = parser.parse_args_and_config()
        assert isinstance(parsed_args, DummyArgs)
        assert parsed_args.model_name == "bert"
        assert parsed_args.learning_rate == 0.001

    def test_parse_multiple_dataclasses(self, monkeypatch):
        parser = CfArgumentParser([DummyArgs, DummyArgsC])
        args = [
            "train",
            "--model_name",
            "bert",
            "--learning_rate",
            "0.001",
            "--C_model",
            "gpt2",
            "--C_learning_rate",
            "0.0005",
        ]
        monkeypatch.setattr(sys, "argv", ["script.py"] + args)

        parsed_args = parser.parse_args_and_config()
        assert isinstance(parsed_args, Tuple)
        assert len(parsed_args) == 2
        assert parsed_args[0].model_name == "bert"
        assert parsed_args[1].C_model == "gpt2"

    def test_parse_yaml_config(self, tmp_path):
        config = {
            "A": {"model": "bert-base", "learning_rate": 0.001},
            "B": {"model": "gpt2", "learning_rate": 0.0001},
            "shared_param": "value",
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        parser = CfArgumentParser([DummyArgsAB])
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

        parser = CfArgumentParser([DummyArgsAB])
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

    def test_yaml_with_defaults(self, monkeypatch, tmp_path):
        config = {"A": {"model": "bert-base"}, "B": {"model": "gpt2"}}

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        parser = CfArgumentParser([DummyArgsAB])
        args = ["train", str(config_path)]
        monkeypatch.setattr(sys, "argv", ["script.py"] + args)

        (parsed_args,) = parser.parse_args_and_config()
        assert parsed_args.A_model == "bert-base"
        assert parsed_args.B_model == "gpt2"
        assert parsed_args.A_learning_rate == 0.001  # default value
        assert parsed_args.B_learning_rate == 0.0001  # default value
        assert parsed_args.shared_param == "default"  # default value

    def test_duplicate_args_in_yaml(self, tmp_path):
        config = {
            "A": {"param": "value1"},
            "A_param": "value2",  # This creates a duplicate A_param
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        parser = CfArgumentParser()
        with pytest.raises(ValueError, match="Duplicate argument"):
            parser._parse_yaml_config(str(config_path))

    def test_yaml_with_invalid_param(self, monkeypatch, tmp_path):
        config = {"invalid_param": "value"}

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        parser = CfArgumentParser([DummyArgs])
        args = ["train", str(config_path)]
        monkeypatch.setattr(sys, "argv", ["script.py"] + args)

        with pytest.raises(SystemExit):  # Printing helpstring
            parser.parse_args_and_config()

    def test_cli_only_with_defaults(self, monkeypatch):
        parser = CfArgumentParser([DummyArgsAB])
        args = ["train", "--A_model", "bert", "--B_model", "gpt2"]
        monkeypatch.setattr(sys, "argv", ["script.py"] + args)

        (parsed_args,) = parser.parse_args_and_config()
        assert parsed_args.A_model == "bert"
        assert parsed_args.B_model == "gpt2"
        assert parsed_args.A_learning_rate == 0.001  # default value
        assert parsed_args.B_learning_rate == 0.0001  # default value
        assert parsed_args.shared_param == "default"  # default value

    def test_task_only_task(self, monkeypatch):
        parser = CfArgumentParser([DummyArgsAB])
        args = ["train"]
        monkeypatch.setattr(sys, "argv", ["script.py"] + args)

        with pytest.raises(SystemExit):
            parser.parse_args_and_config()

    def test_no_task_when_preset(self, monkeypatch):
        parser = CfArgumentParser([DummyArgs], task="train")
        args = ["--model_name", "bert"]
        monkeypatch.setattr(sys, "argv", ["script.py"] + args)

        (parsed_args,) = parser.parse_args_and_config()
        assert parsed_args.model_name == "bert"
        assert parsed_args.learning_rate == 0.001  # default value
