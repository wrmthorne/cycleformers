import sys
from pathlib import Path

from datasets import load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
)

from cycleformers import CycleTrainer, CycleTrainingArguments, DataConfig, ModelConfig
from cycleformers.utils import get_peft_config, suffix_dataclass_factory


def get_model_and_tokenizer(model_config, training_args):
    """Initialize model and tokenizer from config"""
    auto_config = AutoConfig.from_pretrained(model_config.model_name_or_path)
    if not auto_config.is_encoder_decoder:
        model_class = AutoModelForCausalLM
    else:
        model_class = AutoModelForSeq2SeqLM

    model = model_class.from_pretrained(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        trust_remote_code=model_config.trust_remote_code,
        # use_liger_kernel=training_args.use_liger_kernel,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    return model, tokenizer


def main():
    # FIXME: Work out how to get dataclass_A.param_A into form dataclass.param
    parser = HfArgumentParser((CycleTrainingArguments, ModelConfig))

    # FIXME: Allow for cli to override yaml files
    maybe_yaml = Path(sys.argv[1])
    if maybe_yaml.suffix == ".yaml" and maybe_yaml.exists():
        training_args, model_config = parser.parse_yaml_file(maybe_yaml)
    else:
        (
            training_args,
            model_config,
        ) = parser.parse_args_into_dataclasses()

    # TODO: Allow user specified datasests in a generic script
    dataset_A = load_from_disk("./data/en")
    dataset_B = load_from_disk("./data/de")

    # FIXME: Don't force single model and tokenizer
    models, tokenizer = get_model_and_tokenizer(model_config, training_args)
    if not model_config.use_peft:
        model_B, _ = get_model_and_tokenizer(model_config, training_args)
        models = {"A": models, "B": model_B}

    trainer = CycleTrainer(
        args=training_args,
        models=models,
        tokenizers=tokenizer,
        train_dataset_A=dataset_A["train"],
        train_dataset_B=dataset_B["train"],
        eval_dataset_A=dataset_A["test"] if not training_args.eval_strategy == "no" else None,
        eval_dataset_B=dataset_B["test"] if not training_args.eval_strategy == "no" else None,
        peft_configs=get_peft_config(model_config) if model_config.use_peft else None,
    )

    trainer.train()


if __name__ == "__main__":
    main()
