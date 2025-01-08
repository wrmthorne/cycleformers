from pathlib import Path

from datasets import DatasetDict, load_dataset


CURR_DIR = Path(__file__).parent

dataset_configs = {
    "wmt14": {"name": "wmt14", "config": "de-en"},  # Specify language pair
    "conll2003": {"name": "conll2003", "config": None},
}

for dataset_info in dataset_configs.values():
    dataset_dict = DatasetDict()
    for split in ["train[:100]", "validation[:20]", "test[:20]"]:
        # Use streaming to avoid downloading entire dataset
        dataset = load_dataset(
            dataset_info["name"],
            dataset_info["config"],
            split=split,
        )
        dataset_dict[split.split("[")[0]] = dataset

    # Save the small sample dataset
    output_path = CURR_DIR / f"{dataset_info['name']}"
    dataset_dict.save_to_disk(output_path)
