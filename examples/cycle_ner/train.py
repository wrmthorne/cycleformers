from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from cycleformers import (
    CFArgumentParser,
    CycleTrainer,
    CycleTrainingArguments,
    ModelConfig,
    ModelConfigA,
    ModelConfigB,
    merge_configs,
)
from cycleformers.import_utils import is_liger_kernel_available
from cycleformers.task_processors.ner import CONLL2003Processor, CONLL2003ProcessorConfig
from cycleformers.utils import VALID_LIGER_MODELS, get_peft_config


if is_liger_kernel_available():
    from liger_kernel.transformers import AutoLigerKernelForCausalLM


def get_model_and_tokenizer(model_config, training_args):
    """Initialize model and tokenizer from config"""
    config = AutoConfig.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
    )
    config.use_cache = False

    model_kwargs = {}

    if not config.is_encoder_decoder:
        if is_liger_kernel_available() and model_config.use_liger and config.model_type in VALID_LIGER_MODELS:
            model_class = AutoLigerKernelForCausalLM
            model_kwargs["use_liger_kernel"] = training_args.use_liger_kernel
        else:
            model_class = AutoModelForCausalLM
    else:
        model_class = AutoModelForSeq2SeqLM

    model = model_class.from_pretrained(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
        config=config,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        device_map="auto",
    )

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # Print the actual dtype of the first parameter
    print(f"Model weights dtype: {next(model.parameters()).dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    return model, tokenizer


def main():
    parser = CFArgumentParser(
        (CycleTrainingArguments, ModelConfig, ModelConfigA, ModelConfigB, CONLL2003ProcessorConfig), task="train"
    )
    args, model_config_base, model_config_A, model_config_B, conll_config = parser.parse_args_and_config()
    model_config_base = merge_configs(model_config_base, model_config_A, model_config_B)
    args.model_config = model_config_base

    task_processor = CONLL2003Processor(conll_config)
    dataset_A, dataset_B = task_processor.process()

    # Get model A using merged A config
    model_A, tokenizer_A = get_model_and_tokenizer(args.model_config.A, args)

    # Train by adapter swapping
    if not args.use_macct:
        # Get model B using merged B config
        model_B, tokenizer_B = get_model_and_tokenizer(args.model_config.B, args)
        models = {"A": model_A, "B": model_B}
        tokenizers = {"A": tokenizer_A, "B": tokenizer_B} if tokenizer_A != tokenizer_B else tokenizer_A
    else:
        models = model_A
        tokenizers = tokenizer_A

    trainer = CycleTrainer(
        args=args,
        models=models,
        tokenizers=tokenizers,
        train_dataset_A=dataset_A["train"],
        train_dataset_B=dataset_B["train"],
        eval_dataset_A=dataset_A["eval"] if not args.eval_strategy == "no" else None,
        eval_dataset_B=dataset_B["eval"] if not args.eval_strategy == "no" else None,
        peft_configs=get_peft_config(model_config_base),
    )

    trainer.train()


if __name__ == "__main__":
    main()
