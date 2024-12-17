import os
import tempfile

import pytest
from datasets import Dataset, DatasetDict

from cycleformers.task_processors.ner import (
    CONLL2003Processor,
    CONLL2003ProcessorConfig,
    ner_to_sequences,
    reconstruct_sentence,
)


class TestReconstructSentence:
    @pytest.mark.parametrize("token", set(",.!?:;"))
    def test_punctuation_spacing(self, token):
        sample = [token, "A", token, "B", token]
        assert reconstruct_sentence(sample) == f"{token} A{token} B{token}"

    def test_no_space_between_regular_words(self):
        sample = ["A", "B", "C"]
        assert reconstruct_sentence(sample) == "A B C"

    @pytest.mark.parametrize("contraction", set("'s 're 've 'm 't 'll 'd".split()))
    def test_correct_handling_of_contractions(self, contraction):
        sample = ["I", contraction, "A", "B", "C"]
        assert reconstruct_sentence(sample) == f"I{contraction} A B C"

    @pytest.mark.parametrize("quote", set('"' '"'))
    def test_correct_handling_of_quotes(self, quote):
        # FIXME: Currently imperfect handling of quotes but a quick solution without mode imports
        sample = [quote, "A", quote, "B", quote, "C", quote]
        assert reconstruct_sentence(sample) == f"{quote}A{quote}B{quote}C{quote}"


class TestNerToSequences:
    def test_empty_string(self):
        assert ner_to_sequences([], [], "|") == ""

    def test_no_entities(self):
        tokens = ["A", "B", "C"]
        tags = [0, 0, 0]
        sep_token = "|"
        assert ner_to_sequences(tokens, tags, sep_token) == ""

    def test_all_one_entity(self):
        tokens = ["A", "B", "C"]
        tags = [1, 2, 2]
        sep_token = "|"
        assert ner_to_sequences(tokens, tags, sep_token) == "A B C | person"

    def test_all_two_entities(self):
        tokens = ["A", "B", "C"]
        tags = [1, 2, 3]
        sep_token = "|"
        assert ner_to_sequences(tokens, tags, sep_token) == "A B | person | C | organisation"

    def test_missing_B_tags(self):
        tokens = ["A", "B", "C"]
        tags = [2, 2, 2]
        sep_token = "|"
        with pytest.raises(ValueError):
            ner_to_sequences(tokens, tags, sep_token)


class TestCONLL2003Processor:
    @pytest.mark.parametrize(
        "data_dict",
        [
            {
                "train": {"tokens": [["A", "B", "C"]], "ner_tags": [[1, 2, 2]]},
            },
            {
                "train": {"tokens": [["A", "B", "C"]], "ner_tags": [[1, 2, 2]]},
                "test": {"tokens": [["A", "B", "C"]], "ner_tags": [[1, 2, 2]]},
            },
            {
                "train": {"tokens": [["A", "B", "C"]], "ner_tags": [[1, 2, 2]]},
                "val": {"tokens": [["A", "B", "C"]], "ner_tags": [[1, 2, 2]]},
                "test": {"tokens": [["A", "B", "C"]], "ner_tags": [[1, 2, 2]]},
            },
        ],
    )
    def test_preprocess(self, data_dict: dict[str, list]):
        # Ensure that nothing is cached
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.environ["HF_DATASETS_CACHE"] = tmp_dir
            dataset = DatasetDict({split: Dataset.from_dict(data) for split, data in data_dict.items()})

            keys = list(dataset.keys())
            config = CONLL2003ProcessorConfig(cache_dir=tmp_dir)
            processor = CONLL2003Processor()
            dataset_A, dataset_B = processor.preprocess(dataset)
            assert len(dataset_A.keys()) == len(keys)
            assert len(dataset_B.keys()) == len(keys)

            for key in keys:
                assert len(dataset[key]) == len(dataset_A[key]) == len(dataset_B[key])

            keys.remove("train")
            assert "label" not in dataset_A["train"].column_names
            assert "label" not in dataset_B["train"].column_names
            assert dataset_A["train"][0]["text"] != dataset_B["train"][0]["text"]

            for key in keys:
                assert dataset_A[key]["text"] == dataset_B[key]["label"]
                assert dataset_A[key]["label"] == dataset_B[key]["text"]
                assert dataset_A[key]["text"] != dataset_B[key]["text"]
                assert dataset_A[key]["label"] != dataset_B[key]["label"]
