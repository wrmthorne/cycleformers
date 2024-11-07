import transformers
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments

from cycleformers import CycleAdapterConfig, CycleTrainer, PeftCycleModelForSeq2SeqLM
from cycleformers.data import PeftDatasetConfig


transformers.logging.set_verbosity_info()


def download_and_prepare_datasets(tokenizer, num_samples=5000):
    dataset = load_dataset("stas/wmt14-en-de-pre-processed", split="train")
    dataset = dataset.map(
        lambda x: {"english": x["translation"]["en"], "german": x["translation"]["de"]}
    ).remove_columns(["translation"])

    dataset_en = (
        dataset.select_columns("english")
        .shuffle(seed=42)  # Shuffle to simulate not having any exact translations
        .select(range(num_samples))
        .rename_column("english", "text")
        .map(lambda x: {**tokenizer(x["text"]), "train_adapter": "De2En"})
        .remove_columns(["text"])
    )
    dataset_de = (
        dataset.select_columns("german")
        .shuffle(seed=0)  # Shuffle to simulate not having any exact translations
        .select(range(num_samples))
        .rename_column("german", "text")
        .map(lambda x: {**tokenizer(x["text"]), "train_adapter": "En2De"})
        .remove_columns(["text"])
    )
    return dataset_en, dataset_de


def create_model(model_name, model_kwargs={"device_map": "auto"}):
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    lora_config = LoraConfig(task_type="SEQ_2_SEQ_LM", r=32, lora_alpha=64)
    en2de_adapter = CycleAdapterConfig(adapter_name="En2De", peft_config=lora_config)
    de2en_adapter = CycleAdapterConfig(adapter_name="De2En", peft_config=lora_config)

    model = PeftCycleModelForSeq2SeqLM(
        model=base_model, tokenizer=tokenizer, adapter_configs=[en2de_adapter, de2en_adapter]
    )

    return model, tokenizer


# TODO: Develop method for passing config into scripts
if __name__ == "__main__":
    model_name = "google/t5-efficient-tiny"
    model, tokenizer = create_model(model_name)
    dataset_en, dataset_de = download_and_prepare_datasets(tokenizer)

    # Configure datasets with target adapters
    dataset_config_en = PeftDatasetConfig(
        dataset=dataset_en,
        dataset_name="dataset_en",
        train_adapter="De2En",  # Loss(real_en, reconst_en)
    )

    dataset_config_de = PeftDatasetConfig(
        dataset=dataset_de,
        dataset_name="dataset_de",
        train_adapter="En2De",  # Loss(real_de, reconst_de)
    )

    training_args = TrainingArguments(
        output_dir="./outputs",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        save_strategy="epoch",
        # report_to=["wandb"]
    )

    trainer = CycleTrainer(model=model, train_dataset=[dataset_config_en, dataset_config_de], args=training_args)
    trainer.train()
