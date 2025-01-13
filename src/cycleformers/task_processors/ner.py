from collections.abc import Callable
from dataclasses import dataclass
from itertools import zip_longest
from typing import Literal

import evaluate
from datasets import DatasetDict

from cycleformers.cycle_trainer_utils import EvalGeneration

from .base import BaseProcessor, ProcessorConfig


tag_to_idx = {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4, "B-LOC": 5, "I-LOC": 6, "B-MISC": 7, "I-MISC": 8}
idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}
tag_to_string = {"PER": "person", "ORG": "organisation", "LOC": "location", "MISC": "miscellaneous"}
string_to_tag = {string: tag for tag, string in tag_to_string.items()}


@dataclass
class CONLL2003ProcessorConfig(ProcessorConfig):
    """Configuration class for CONLL2003 dataset processor.

    This class extends the base ProcessorConfig with CONLL2003-specific parameters.

    Args:
        dataset_name (str): HuggingFace dataset name/path. Defaults to "eriktks/conll2003".
        preset (Literal["entity_seqs"] | None): Processing preset to use. Currently only supports "entity_seqs"
            which extracts sequences containing named entities. Defaults to "entity_seqs".
        sep_token (str): Token used to separate elements in processed sequences. Defaults to "|".

    Example:
        >>> config = CONLL2003ProcessorConfig(
        ...     dataset_name="conll2003",
        ...     preset="entity_seqs",
        ...     sep_token="|"
        ... )
        >>> processor = CONLL2003Processor(config)
    """

    dataset_name: str = "eriktks/conll2003"
    preset: Literal["entity_seqs"] | None = "entity_seqs"
    # TODO: work out how to allow this param to multiple dataclasses in CLI
    # trust_remote_code: bool = field(
    #     default=True,
    #     metadata={"help": "Must be enabled to load the dataset", "dest": "trust-remote-code"}
    # )
    sep_token: str = "|"


def reconstruct_sentence(tokens: list[str]) -> str:
    """Reconstructs a sentence from CoNLL-2003 tokens while handling whitespace correctly.

    This function takes a list of tokens from the CoNLL-2003 format and reconstructs them into a properly formatted sentence
    by applying standard English language spacing rules:

    - No spaces before punctuation marks (,.!?:;)
    - Spaces between regular words
    - Proper handling of contractions (don't, I'm, etc.)
    - Correct spacing around quotation marks

    Args:
        tokens (list[str]): List of tokens from the CoNLL-2003 dataset

    Returns:
        str: The reconstructed sentence with proper spacing and punctuation

    Example:
        >>> tokens = ["I", "don", "'t", "like", "New", "York", "."]
        >>> reconstruct_sentence(tokens)
        "I don't like New York."
    """
    if not tokens:
        return ""

    # Define punctuation categories
    closing_puncts = set(",.!?:;")
    opening_quotes = set('"' '"')
    closing_quotes = set('"' '"')
    contractions = set("'s 're 've 'm 't 'll 'd".split())

    result = []
    for i, token in enumerate(tokens):
        if i == 0:
            result.append(token)
            continue

        prev_token = tokens[i - 1]

        needs_space = not (
            token in closing_puncts
            or token.lower() in contractions
            or prev_token in opening_quotes
            or token in closing_quotes
        )

        if needs_space:
            result.append(" " + token)
        else:
            result.append(token)

    return "".join(result)


def bio_to_entity_seq(tokens: list[str], tags: list[int], sep_token: str) -> str:
    """Convert a list of tokens and their corresponding tags to a sequence of entity types.

    This function takes tokens and their NER tags and converts them into a sequence format
    suitable for sequence-to-sequence training. It follows the BIO2 tagging scheme where:
    - B- prefix indicates beginning of an entity
    - I- prefix indicates inside/continuation of an entity
    - O indicates outside any entity (not used in output)

    Args:
        tokens (list[str]): List of string tokens from a single sentence
        tags (list[int]): List of integer tags corresponding to the tokens using BIO2 scheme
        sep_token (str): Separator token to use between entity and its type

    Returns:
        str: A string containing entities and their types separated by the sep_token.
            For example: "John Smith | person Google | organization"

    Raises:
        ValueError: If an I- tag appears without a preceding B- tag (invalid BIO2 sequence)

    Example:
        >>> tokens = ["John", "Smith", "works", "at", "Google", "."]
        >>> tags = [1, 2, 0, 0, 7, 0]  # B-PER, I-PER, O, O, B-ORG, O
        >>> ner_to_sequences(tokens, tags, " | ")
        'John Smith | person Google | organization'
    """
    compound_tokens = []
    for token, tag in zip(tokens, tags):
        if tag in [1, 3, 5, 7]:
            tag_string = tag_to_string[idx_to_tag[tag].split("-")[-1]]
            compound_tokens.append([token, sep_token, tag_string])
        elif tag in [2, 4, 6, 8]:
            if not compound_tokens:
                raise ValueError("Missing B-tag before I-tag. Please use BIO2 tagging scheme, not BIO1.")

            compound_tokens[-1].insert(-2, token)

    return f" {sep_token} ".join([" ".join(token_tags) for token_tags in compound_tokens])


def format_tag_sequence(tags):
    formatted = []
    prev_tag = None
    for tag in tags:
        if tag == "O":
            formatted.append("O")
            prev_tag = None
        elif tag != prev_tag:
            formatted.append(f"B-{tag}")
            prev_tag = tag
        else:
            formatted.append(f"I-{tag}")
    return formatted


def aligned_entity_seqs_to_tags(
    preds: str | list[str], labels: str | list[str], sep_token: str
) -> tuple[list[str], list[str]]:
    """Convert aligned entity sequences to BI tags."""
    # Handle empty sequences
    if not preds or not labels:
        return ["O"], ["O"]

    # Handle both single strings and lists
    if isinstance(preds, list):
        all_pred_tags = []
        all_label_tags = []
        for pred, label in zip(preds, labels):
            if isinstance(label, str):
                pred_tags, label_tags = aligned_entity_seqs_to_tags(pred, label, sep_token)
                all_pred_tags.extend(pred_tags)
                all_label_tags.extend(label_tags)
        # Handle case where we got no tags
        if not all_pred_tags:
            return ["O"], ["O"]
        return all_pred_tags, all_label_tags

    # At this point, mypy knows preds must be str
    assert isinstance(preds, str)
    assert isinstance(labels, str)

    preds_parts = [[token.strip() for token in part.lower().strip().split()] for part in preds.split(sep_token)]
    labels_parts = [[token.strip() for token in part.lower().strip().split()] for part in labels.split(sep_token)]

    preds_tags = []
    labels_tags = []
    for p_toks, p_tags, l_toks, l_tags in zip(
        preds_parts[::2], preds_parts[1::2], labels_parts[::2], labels_parts[1::2]
    ):
        p_tags += [p_tags[-1]] * (len(p_toks) - len(p_tags))
        l_tags += [l_tags[-1]] * (len(l_toks) - len(l_tags))

        zipped = list(zip_longest(p_toks, p_tags, l_toks, l_tags, fillvalue="O"))
        _, p_tags_new, _, l_tags_new = map(list, zip(*zipped))

        preds_tags.extend(format_tag_sequence(p_tags_new))
        labels_tags.extend(format_tag_sequence(l_tags_new))

    return preds_tags, labels_tags


def maybe_valid_sentence_to_parts(sentence: str, sep_token: str) -> list[str]:
    if not sentence or sep_token not in sentence:
        return []
    return [p.strip() for p in sentence.split(sep_token)]


class CONLL2003Processor(BaseProcessor):
    """Processor for the CONLL2003 Named Entity Recognition dataset.

    This processor converts NER data between raw text and entity sequence formats for cycle training.
    It supports both directions:
        - Text -> Entity sequences (e.g. "John works at Google" -> "John | person Google | organization")
        - Entity sequences -> Text (e.g. "John | person Google | organization" -> "John works at Google")

    Args:
        config (`CONLL2003ProcessorConfig`, *optional*):
            The configuration controlling processor behavior. Includes settings like separator token
            between entities and their types. Defaults to `CONLL2003ProcessorConfig()`.

    The processor handles:
        - Loading and preprocessing CONLL2003-style datasets
        - Converting token-level NER annotations to sequence format
        - Creating complementary datasets for cycle training
        - Computing evaluation metrics using seqeval

    Example:
        >>> from cycleformers.task_processors import CONLL2003Processor
        >>> from cycleformers.task_processors.ner import CONLL2003ProcessorConfig
        >>>
        >>> config = CONLL2003ProcessorConfig(sep_token=" | ")
        >>> processor = CONLL2003Processor(config)
        >>> dataset_A, dataset_B = processor.process()
        >>> print(dataset_A["train"][0])
        {'text': 'John Smith works at Google.'}
        >>> print(dataset_B["train"][0])
        {'text': 'John Smith | person Google | organization'}

    For more details on NER processors and their configurations, see the
    [documentation](https://wrmthorne.github.io/cycleformers/conceptual_reference/task_processors).
    """

    def __init__(self, config: CONLL2003ProcessorConfig = CONLL2003ProcessorConfig()):
        super().__init__(config)
        # Ensure formatting of sep token is correct
        self.config: CONLL2003ProcessorConfig = config  # type annotation for config
        self.sep_token = config.sep_token.strip()

    # Computes sentence to entity sequence metrics (dataset A)
    def compute_metrics_A(self, eval_pred: EvalGeneration) -> dict[str, float]:
        """Compute metrics for NER task."""
        metrics = evaluate.load("seqeval")

        # Convert predictions and labels to aligned BI tags
        predictions, references = aligned_entity_seqs_to_tags(eval_pred.predictions, eval_pred.labels, self.sep_token)

        # Wrap predictions and references in lists since seqeval expects lists of sequences
        predictions_list: list[list[str]] = [predictions]
        references_list: list[list[str]] = [references]

        return metrics.compute(predictions=predictions_list, references=references_list, zero_division=0)

    def compute_metrics(self) -> dict[str, Callable[[EvalGeneration], dict[str, float]]]:
        return {"A": self.compute_metrics_A}

    def preprocess(self, dataset: DatasetDict) -> tuple[DatasetDict, DatasetDict]:
        original_cols = dataset["train"].column_names
        dataset = dataset.map(
            lambda x: {
                "sentence": reconstruct_sentence(x["tokens"]),
                "entity_seq": bio_to_entity_seq(x["tokens"], x["ner_tags"], self.sep_token),
            }
        ).remove_columns(original_cols)

        dataset_A = dataset.map(lambda x: {"text": x["sentence"], "labels": x["entity_seq"]})
        dataset_B = dataset.map(lambda x: {"text": x["entity_seq"], "labels": x["sentence"]})

        dataset_A["train"] = dataset_A["train"].remove_columns(["labels"])
        dataset_B["train"] = dataset_B["train"].remove_columns(["labels"])

        return dataset_A, dataset_B
