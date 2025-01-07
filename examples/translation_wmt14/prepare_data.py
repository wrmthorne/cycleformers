from pathlib import Path

from datasets import DatasetDict, load_dataset


def prepare_wmt14_en_de_datasets(num_samples=5000, test_size=100):
    dataset = load_dataset("wmt/wmt14", split="train")
    dataset = dataset.map(
        lambda x: {"english": x["translation"]["en"], "german": x["translation"]["de"]}
    ).remove_columns(["translation"])

    # Pull out sample of paired data for evaluation
    dataset = dataset.train_test_split(test_size=test_size)

    en_train = (
        dataset["train"]
        .select_columns("english")
        .shuffle(seed=42)  # Shuffle to simulate not having any exact translations
        .select(range(num_samples))
        .rename_column("english", "text")
    )
    en_eval = dataset["test"].rename_columns({"english": "text", "german": "labels"})
    dataset_en = DatasetDict({"train": en_train, "test": en_eval})

    de_train = (
        dataset["train"]
        .select_columns("german")
        .shuffle(seed=0)  # Shuffle to simulate not having any exact translations
        .select(range(num_samples))
        .rename_column("german", "text")
    )
    de_eval = dataset["test"].rename_columns({"german": "text", "english": "labels"})
    dataset_de = DatasetDict({"train": de_train, "test": de_eval})

    return dataset_en, dataset_de


if __name__ == "__main__":
    dataset_en, dataset_de = prepare_wmt14_en_de_datasets()

    Path("./data").mkdir(exist_ok=True)

    dataset_en.save_to_disk("./data/en")
    dataset_de.save_to_disk("./data/de")
