import sys
from pathlib import Path

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, HfArgumentParser

from cycleformers import CycleTrainer, CycleTrainingArguments, ModelConfig
from cycleformers.import_utils import is_liger_available
from cycleformers.task_processors.ner import CONLL2003Processor, CONLL2003ProcessorConfig
from cycleformers.utils import get_peft_config


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
        # use_liger_kernel=training_args.use_liger_kernel and is_liger_available(),
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    return model, tokenizer


def main():
    # FIXME: Work out how to get dataclass_A.param_A into form dataclass.param
    parser = HfArgumentParser((CycleTrainingArguments, ModelConfig, CONLL2003ProcessorConfig))

    maybe_yaml = Path(sys.argv[1])
    if maybe_yaml.suffix == ".yaml" and maybe_yaml.exists():
        training_args, model_config, conll_config = parser.parse_yaml_file(maybe_yaml)
    else:
        raise ValueError("Only support for yaml right now")

    task_processor = CONLL2003Processor(conll_config)
    dataset_A, dataset_B = task_processor.process()

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
        eval_dataset_A=dataset_A["eval"] if not training_args.eval_strategy == "no" else None,
        eval_dataset_B=dataset_B["eval"] if not training_args.eval_strategy == "no" else None,
        peft_configs=get_peft_config(model_config),
    )

    trainer.train()


if __name__ == "__main__":
    main()
