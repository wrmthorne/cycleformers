import json
import sys
from collections.abc import Iterable

import yaml
from transformers.hf_argparser import DataClassType, HfArgumentParser


VALID_TASKS = ["train"]


class CfArgumentParser(HfArgumentParser):
    def __init__(
        self,
        dataclass_types: list[DataClassType] | None = None,
        task: str | None = None,
        **kwargs,
    ):
        # Make sure dataclass_types is an iterable
        if dataclass_types is None:
            dataclass_types = []
        elif not isinstance(dataclass_types, Iterable):
            dataclass_types = [dataclass_types]

        self.task = task
        super().__init__(dataclass_types=dataclass_types, **kwargs)

    def _parse_yaml_config(self, config_file: str):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        # Convert config YAMLs to be A_<key> and B_<key>
        config_a = {f"A_{k}": v for k, v in config.pop("A", {}).items()}
        config_b = {f"B_{k}": v for k, v in config.pop("B", {}).items()}

        file_args = []
        for c in [config, config_a, config_b]:
            for k, v in c.items():
                v = json.dumps(v) if isinstance(v, dict) else str(v)

                if f"--{k}" in file_args:
                    raise ValueError(f"Duplicate argument {k} found in config files")

                file_args.extend([f"--{k}", v])
        return file_args

    def parse_args_and_config(self):
        args = sys.argv[1:]

        # Handle task argument
        if not self.task and len(args) == 0:
            raise ValueError(f"No task provided. Task must be one of {VALID_TASKS}.")

        # Task is already set
        if self.task and args[0] in VALID_TASKS:
            raise ValueError(f"Task already set by script to {self.task}. Try again without {args[0]}.")
        # Task is not set and arg is a valid task
        elif not self.task:
            if args[0] in VALID_TASKS:
                self.task = args.pop(0)
            else:
                raise ValueError(f"Task must be one of {VALID_TASKS}, got {args[0]}")

        if len(args) > 0 and (args[0].endswith(".yaml") or args[0].endswith(".yml")):
            config_file = args.pop(0)
            file_args = self._parse_yaml_config(config_file)
            args = file_args + args

        return self.parse_args_into_dataclasses(args)
