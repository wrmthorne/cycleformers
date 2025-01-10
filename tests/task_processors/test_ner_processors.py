"""
AWAITING REFACTOR
"""

from pathlib import Path

from cycleformers.task_processors.ner import CONLL2003Processor, CONLL2003ProcessorConfig


SAMPLES_DIR = Path(__file__).parent.parent / "sample_data"


class TestCONLL2003Processor:
    def test_preprocess(self, monkeypatch, temp_dir):
        """Test preprocessing of CONLL2003 dataset."""
        monkeypatch.setenv("HF_DATASETS_CACHE", str(temp_dir))

        config = CONLL2003ProcessorConfig(
            dataset_name=SAMPLES_DIR / "conll2003",
            cache_dir=str(temp_dir),
            dataset_seed=42,  # Fixed seed for reproducible tests
        )
        processor = CONLL2003Processor(config)
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
        config2 = CONLL2003ProcessorConfig(
            dataset_name=SAMPLES_DIR / "conll2003",
            dataset_seed=42,
            cache_dir=str(temp_dir),
        )
        processor2 = CONLL2003Processor(config2)
        dataset_A2, dataset_B2 = processor2.process()

        assert list(dataset_A["train"]["text"]) == list(dataset_A2["train"]["text"])
        assert list(dataset_B["train"]["text"]) == list(dataset_B2["train"]["text"])

        # Check that evaluation splits maintain parallel data
        for key in ["validation", "test"]:
            assert dataset_A[key]["text"] == dataset_B[key]["label"]
            assert dataset_A[key]["label"] == dataset_B[key]["text"]
            assert dataset_A[key]["text"] != dataset_B[key]["text"]
            assert dataset_A[key]["label"] != dataset_B[key]["label"]
