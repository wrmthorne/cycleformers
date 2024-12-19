import json
import sys

import yaml
from transformers.hf_argparser import DataClassType, HfArgumentParser


TASKS = ["train"]


class CFArgumentParser(HfArgumentParser):
    def __init__(
        self,
        dataclass_types: list[DataClassType] | None = None,
        task: str | None = None,
        **kwargs,
    ):
        self.task = task
        super().__init__(dataclass_types, **kwargs)

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

                if k in file_args:
                    raise ValueError(f"Duplicate argument {k} found in config files")

                file_args.extend([f"--{k}", v])
        return file_args

    def parse_args_and_config(self):
        args = sys.argv[1:]

        if self.task is not None and args[0] in TASKS and not args[0] == self.task:
            raise ValueError(f"Task must be one of {TASKS}, got {self.task}")
        elif self.task is not None and args[0] in TASKS and args[0] == self.task:
            args.pop(0)
        elif self.task is None and args[0] in TASKS:
            self.task = args.pop(0)
        elif self.task is None and args[0] not in TASKS:
            raise ValueError(f"Task must be one of {TASKS}, got {args[0]}")

        if args[0].endswith(".yaml") or args[0].endswith(".yml"):
            config_file = args.pop(0)
            file_args = self._parse_yaml_config(config_file)
            args = file_args + args

        return self.parse_args_into_dataclasses(args)
