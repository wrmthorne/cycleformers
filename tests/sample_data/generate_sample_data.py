from pathlib import Path

from datasets import DatasetDict, load_dataset


CURR_DIR = Path(__file__).parent

dataset_configs = {
    "wmt14": ["wmt/wmt14", "de-en"],
    "conll2003": ["eriktks/conll2003"],
}

for dataset_name, dataset_info in dataset_configs.items():
    dataset_dict = DatasetDict()
    for split in ["train[:100]", "validation[:20]", "test[:20]"]:
        # Use streaming to avoid downloading entire dataset
        dataset = load_dataset(
            *dataset_info,
            split=split,
        )
        dataset_dict[split.split("[")[0]] = dataset

    print(dataset_dict["train"][0])

    # Save the small sample dataset
    output_path = CURR_DIR / f"{dataset_name}"
    dataset_dict.save_to_disk(output_path)
