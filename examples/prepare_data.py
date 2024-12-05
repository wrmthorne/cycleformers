from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict


def download_and_prepare_datasets(tokenizer, num_samples=5000, test_size=100):
    dataset = load_dataset("stas/wmt14-en-de-pre-processed", split="train")
    dataset = dataset.map(
        lambda x: {"english": x["translation"]["en"], "german": x["translation"]["de"]}
    ).remove_columns(["translation"])

    # Pull out sample of paired data for evaluation
    dataset = dataset.train_test_split(test_size=test_size)

    en_train = (
        dataset["train"].select_columns("english")
        .shuffle(seed=42)  # Shuffle to simulate not having any exact translations
        .select(range(num_samples))
        .rename_column("english", "text")
        .map(lambda x: {**tokenizer(x["text"])})
        .remove_columns(["text"])
    )
    en_eval = (
        dataset["test"]
        .rename_columns({"english": "text", "german": "labels"})
        .map(lambda x: {**tokenizer(x["text"]), "labels": tokenizer.encode(x["labels"])})
        .remove_columns(["text"])
    )
    dataset_en = DatasetDict({"train": en_train, "test": en_eval})

    de_train = (
        dataset["train"].select_columns("german")
        .shuffle(seed=0)  # Shuffle to simulate not having any exact translations
        .select(range(num_samples))
        .rename_column("german", "text")
        .map(lambda x: {**tokenizer(x["text"])})
        .remove_columns(["text"])
    )
    de_eval = (
        dataset["test"]
        .rename_columns({"german": "text", "english": "labels"})
        .map(lambda x: {**tokenizer(x["text"]), "labels": tokenizer.encode(x["labels"])})
        .remove_columns(["text"])
    )
    dataset_de = DatasetDict({"train": de_train, "test": de_eval})

    return dataset_en, dataset_de


if __name__ == "__main__":
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset_en, dataset_de = download_and_prepare_datasets(tokenizer)

    Path("data").mkdir(exist_ok=True)

    dataset_en.save_to_disk("data/en")
    dataset_de.save_to_disk("data/de")

