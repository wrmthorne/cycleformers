from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

from cycleformers.task_processors.translation import TranslationProcessor, TranslationProcessorConfig


SAMPLES_DIR = Path(__file__).parent.parent / "sample_data"


class TestTranslationProcessor:
    def test_preprocess_wmt_format(self, monkeypatch, temp_dir):
        """Test preprocessing of WMT-style datasets with nested translation dictionaries."""
        monkeypatch.setenv("HF_DATASETS_CACHE", str(temp_dir))

        config = TranslationProcessorConfig(
            dataset_name=SAMPLES_DIR / "wmt14",
            dataset_config_name="de-en",  # Required for WMT14
            dataset_seed=42,  # Fixed seed for reproducible tests
            cache_dir=str(temp_dir),
        )
        processor = TranslationProcessor(config)
        dataset_A, dataset_B = processor.process()

        # Check that all splits are preserved
        assert set(dataset_A.keys()) == {"train", "validation", "test"}
        assert set(dataset_B.keys()) == {"train", "validation", "test"}

        # Check that training data is non-parallel (no labels)
        assert "label" not in dataset_A["train"].column_names
        assert "label" not in dataset_B["train"].column_names

        # Verify training splits are properly shuffled and unaligned
        train_texts_A = dataset_A["train"]["text"]
        train_texts_B = dataset_B["train"]["text"]

        # Check no exact alignment exists
        assert not any(
            all(a == b for a, b in zip(train_texts_A, train_texts_B[i:] + train_texts_B[:i]))
            for i in range(len(train_texts_B))
        ), "Training splits appear to be aligned with some offset"

        # Verify that shuffling is deterministic with the same seed
        config2 = TranslationProcessorConfig(
            dataset_name=SAMPLES_DIR / "wmt14",
            dataset_config_name="de-en",
            dataset_seed=42,
            cache_dir=str(temp_dir),
        )
        processor2 = TranslationProcessor(config2)
        dataset_A2, dataset_B2 = processor2.process()

        assert list(dataset_A["train"]["text"]) == list(dataset_A2["train"]["text"])
        assert list(dataset_B["train"]["text"]) == list(dataset_B2["train"]["text"])

        # Check that evaluation splits maintain parallel data
        for key in ["validation", "test"]:
            assert dataset_A[key]["text"] == dataset_B[key]["label"]
            assert dataset_A[key]["label"] == dataset_B[key]["text"]
            assert dataset_A[key]["text"] != dataset_B[key]["text"]
            assert dataset_A[key]["label"] != dataset_B[key]["label"]

    def test_preprocess_flat_format(self):
        """Test preprocessing of datasets with flat source/target columns."""
        data = {
            "train": {
                "source": ["Hello", "World", "Good", "Morning"],
                "target": ["Hallo", "Welt", "Gut", "Morgen"],
            },
            "test": {
                "source": ["Test"],
                "target": ["Test"],
            },
        }
        dataset = DatasetDict({split: Dataset.from_dict(data) for split, data in data.items()})

        config = TranslationProcessorConfig(
            dataset_name=SAMPLES_DIR / "wmt14",
            source_column="source",
            target_column="target",
            dataset_seed=42,  # Fixed seed for reproducible tests
        )
        processor = TranslationProcessor(config)
        dataset_A, dataset_B = processor.preprocess(dataset)

        # Check basic structure
        assert len(dataset_A.keys()) == len(dataset.keys())
        assert len(dataset_B.keys()) == len(dataset.keys())

        # Verify training splits are properly shuffled and unaligned
        train_texts_A = dataset_A["train"]["text"]
        train_texts_B = dataset_B["train"]["text"]

        # Check no exact alignment exists
        assert not any(
            all(a == b for a, b in zip(train_texts_A, train_texts_B[i:] + train_texts_B[:i]))
            for i in range(len(train_texts_B))
        ), "Training splits appear to be aligned with some offset"

        # Verify that shuffling is deterministic with the same seed
        config2 = TranslationProcessorConfig(
            source_column="source",
            target_column="target",
            dataset_seed=42,
        )
        processor2 = TranslationProcessor(config2)
        dataset_A2, dataset_B2 = processor2.preprocess(dataset)

        assert list(dataset_A["train"]["text"]) == list(dataset_A2["train"]["text"])
        assert list(dataset_B["train"]["text"]) == list(dataset_B2["train"]["text"])

        # Check evaluation data is parallel
        assert dataset_A["test"][0]["text"] == "Test"
        assert dataset_A["test"][0]["label"] == "Test"

    def test_custom_preprocessing(self):
        """Test preprocessing with a custom preprocessing function."""
        data = {
            "train": {
                "text": [
                    "en: Hello || de: Hallo",
                    "en: World || de: Welt",
                    "en: Good || de: Gut",
                    "en: Morning || de: Morgen",
                ],
            },
            "test": {
                "text": ["en: Test || de: Test"],
            },
        }
        dataset = DatasetDict({split: Dataset.from_dict(data) for split, data in data.items()})

        def custom_preprocessor(example):
            en, de = example["text"].split("||")
            return {
                "source": en.split(": ")[1].strip(),
                "target": de.split(": ")[1].strip(),
            }

        config = TranslationProcessorConfig(
            dataset_name=SAMPLES_DIR / "wmt14",
            preprocessing_fn=custom_preprocessor,
            dataset_seed=42,  # Fixed seed for reproducible tests
        )
        processor = TranslationProcessor(config)
        dataset_A, dataset_B = processor.preprocess(dataset)

        # Verify training splits are properly shuffled and unaligned
        train_texts_A = dataset_A["train"]["text"]
        train_texts_B = dataset_B["train"]["text"]

        # Check no exact alignment exists
        assert not any(
            all(a == b for a, b in zip(train_texts_A, train_texts_B[i:] + train_texts_B[:i]))
            for i in range(len(train_texts_B))
        ), "Training splits appear to be aligned with some offset"

        # Verify that shuffling is deterministic with the same seed
        config2 = TranslationProcessorConfig(
            preprocessing_fn=custom_preprocessor,
            dataset_seed=42,
        )
        processor2 = TranslationProcessor(config2)
        dataset_A2, dataset_B2 = processor2.preprocess(dataset)

        assert list(dataset_A["train"]["text"]) == list(dataset_A2["train"]["text"])
        assert list(dataset_B["train"]["text"]) == list(dataset_B2["train"]["text"])

        # Check evaluation data is parallel
        assert dataset_A["test"][0]["text"] == "Test"
        assert dataset_A["test"][0]["label"] == "Test"
