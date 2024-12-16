from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

from datasets import DatasetDict, load_dataset

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
    # trust_remote_code: bool = False
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


def ner_to_sequences(tokens: list[str], tags: list[int], sep_token: str) -> str:
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


class CONLL2003Processor(BaseProcessor):
    """Processor for the CONLL2003 Named Entity Recognition dataset.

    This processor handles the CONLL2003 dataset which contains text annotated with named entities.
    It converts the dataset into two formats:
    - Dataset A: Raw text -> Entity sequences
    - Dataset B: Entity sequences -> Raw text

    The processor:
    1. Loads the CONLL2003 dataset
    2. Converts the token-level NER annotations into sequence format
    3. Creates two complementary datasets for cycle training

    Args:
        config (CONLL2003ProcessorConfig): Configuration object controlling processor behavior.
            Includes settings like separator token between entities and their types.

    Example:
        >>> config = CONLL2003ProcessorConfig(sep_token=" | ")
        >>> processor = CONLL2003Processor(config)
        >>> dataset_A, dataset_B = processor.process()
        >>> print(dataset_A["train"][`0`])
        {'text': 'John Smith works at Google.'}
        >>> print(dataset_B["train"][`0`])
        {'text': 'John Smith | person Google | organization'}
    """

    def __init__(self, config: CONLL2003ProcessorConfig = CONLL2003ProcessorConfig()):
        super().__init__(config)
        # Ensure formatting of sep token is correct
        self.config: CONLL2003ProcessorConfig = config  # type annotation for config
        self.sep_token = config.sep_token.strip()

    def load(self) -> DatasetDict:
        return load_dataset(
            self.config.dataset_name,
            trust_remote_code=True,  # FIXME: Don't allow by default or hardcode
            cache_dir=self.config.cache_dir,
        )

    def preprocess(self, dataset: DatasetDict) -> tuple[DatasetDict, DatasetDict]:
        original_cols = dataset["train"].column_names
        dataset_A = dataset.map(
            lambda x: {
                "sentence": reconstruct_sentence(x["tokens"]),
                "entity_seq": ner_to_sequences(x["tokens"], x["ner_tags"], self.sep_token),
            }
        ).remove_columns(original_cols)

        dataset_B = deepcopy(dataset_A)
        dataset_A = dataset_A.rename_columns({"sentence": "text", "entity_seq": "label"})
        dataset_A["train"] = dataset_A["train"].remove_columns(["label"])
        dataset_B = dataset_B.rename_columns({"entity_seq": "text", "sentence": "label"})
        dataset_B["train"] = dataset_B["train"].remove_columns(["label"])

        return dataset_A, dataset_B
