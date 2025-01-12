"""
AWAITING REFACTOR
"""

from pathlib import Path

import pytest

from cycleformers.cycle_trainer_utils import EvalGeneration
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
        assert "labels" not in dataset_A["train"].column_names
        assert "labels" not in dataset_B["train"].column_names

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
            assert dataset_A[key]["text"] == dataset_B[key]["labels"]
            assert dataset_A[key]["labels"] == dataset_B[key]["text"]
            assert dataset_A[key]["text"] != dataset_B[key]["text"]
            assert dataset_A[key]["labels"] != dataset_B[key]["labels"]

    @pytest.mark.parametrize(
        "test_case, predictions, labels, expected_f1, expected_accuracy",
        [
            (
                "perfect_match",
                ["John Smith | person | Google | organization"],
                ["John Smith | person | Google | organization"],
                1.0,
                1.0,
            ),
            (
                "completely_wrong",
                ["John Smith | organization | Google | person"],
                ["John Smith | person | Google | organization"],
                0.0,
                None,  # We don't check accuracy for wrong predictions
            ),
            (
                "partial_match",
                ["John Smith | person | Google | person"],
                ["John Smith | person | Google | organization"],
                pytest.approx(0.5, abs=0.5),  # Should be between 0 and 1
                None,
            ),
            ("empty_prediction", [""], ["John Smith | person"], 0.0, None),
            (
                "invalid_format",
                ["Invalid format", "Just text | invalid"],
                ["John Smith | person", "Google | organization"],
                0.0,
                None,
            ),
            (
                "multiple_entities",
                [
                    "John Smith | person | Google | organization | New York | location",
                    "Microsoft | organization | Apple | organization",
                ],
                [
                    "John Smith | person | Google | organization | New York | location",
                    "Microsoft | organization | Apple | organization",
                ],
                1.0,
                1.0,
            ),
        ],
    )
    def test_calculate_metrics_A(self, test_case, predictions, labels, expected_f1, expected_accuracy):
        """Test calculation of metrics for NER task with different scenarios."""
        config = CONLL2003ProcessorConfig(sep_token=" | ")
        processor = CONLL2003Processor(config)

        eval_pred = EvalGeneration(predictions=predictions, labels=labels)
        metrics = processor.compute_metrics_A(eval_pred)

        assert metrics["overall_f1"] == expected_f1
        if expected_accuracy is not None:
            assert metrics["overall_accuracy"] == expected_accuracy

    # TODO: Test compute_metrics
