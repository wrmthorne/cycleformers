from collections.abc import Callable
from dataclasses import dataclass

import evaluate
from datasets import DatasetDict, IterableDatasetDict

from cycleformers.cycle_trainer_utils import EvalGeneration

from .base import BaseProcessor, ProcessorConfig


# Create a single metricc object with sacrebleu, chrf, meteor, bertscore and rouge
# bertscore = evaluate.load("bertscore")
# bertscore.compute = partial(bertscore.compute, lang="en")

metrics = {
    "sacrebleu": evaluate.load("sacrebleu"),
    "rouge": evaluate.load("rouge"),
    # "meteor": evaluate.load("meteor"),
    # "bertscore": bertscore,
}

eval_cls = evaluate.combine(metrics, force_prefix=True)


@dataclass
class TranslationProcessorConfig(ProcessorConfig):
    """Configuration class for translation dataset processors.

    This class extends the base ProcessorConfig with translation-specific parameters.

    Args:
        dataset_name (str): HuggingFace dataset name/path. Defaults to "wmt14".
        dataset_config_name (str | None): Specific configuration of the dataset to load.
            For WMT14, must be one of ['cs-en', 'de-en', 'fr-en', 'hi-en', 'ru-en'].
            Defaults to "de-en".
        source_lang (str): Source language code (e.g., "en" for English). Defaults to "en".
        target_lang (str): Target language code (e.g., "de" for German). Defaults to "de".
        source_column (str): Column name containing source text. Defaults to "translation".
        target_column (str): Column name containing target text. If None, uses source_column. Defaults to None.
        preprocessing_fn (callable | None): Optional function to preprocess raw dataset entries.
            Should take a dataset entry and return a dict with 'source' and 'target' keys.
            Defaults to None.

    Example:
        >>> config = TranslationProcessorConfig(
        ...     dataset_name="wmt14",
        ...     dataset_config_name="de-en",
        ...     source_lang="en",
        ...     target_lang="de"
        ... )
        >>> processor = TranslationProcessor(config)
    """

    dataset_name: str = "wmt14"
    dataset_config_name: str = "de-en"  # Required for WMT14
    source_lang: str = "en"
    target_lang: str = "de"
    source_column: str = "translation"
    target_column: str | None = None
    preprocessing_fn: Callable | None = None


class TranslationProcessor(BaseProcessor):
    """Processor for machine translation datasets.

    This processor handles translation datasets in various formats and converts them into cycleformers-compatible format.
    It supports both standard parallel corpora (source -> target) and back-translation style training (target -> source).

    Args:
        config (`TranslationProcessorConfig`, *optional*):
            The configuration controlling processor behavior. Includes settings like dataset name,
            language pairs, and column names. Defaults to `TranslationProcessorConfig()`.

    The processor handles different dataset formats:
        - Nested dictionary format (e.g. WMT datasets): `{'translation': {'en': '...', 'de': '...'}}`
        - Flat dictionary format: `{'source': '...', 'target': '...'}`
        - Custom formats via `preprocessing_fn`

    Example:
        >>> from cycleformers.task_processors import TranslationProcessor
        >>> from cycleformers.task_processors.translation import TranslationProcessorConfig
        >>>
        >>> config = TranslationProcessorConfig(
        ...     dataset_name="wmt14",
        ...     dataset_config_name="de-en",
        ...     source_lang="en",
        ...     target_lang="de"
        ... )
        >>> processor = TranslationProcessor(config)
        >>> dataset_A, dataset_B = processor.process()
        >>> print(dataset_A["train"][0])
        {'text': 'The cat sat on the mat.'}
        >>> print(dataset_B["train"][0])
        {'text': 'Die Katze saß auf der Matte.'}

    For more details on translation processors and their configurations, see the
    [documentation](https://wrmthorne.github.io/cycleformers/conceptual_reference/task_processors).
    """

    def __init__(self, config: TranslationProcessorConfig = TranslationProcessorConfig()):
        super().__init__(config)
        self.config: TranslationProcessorConfig = config

    def _extract_text_pair(self, example: dict) -> dict:
        """Extract source and target text from a dataset example.

        This method handles different dataset formats:
        1. Nested dictionary format (e.g., {'translation': {'en': '...', 'de': '...'}})
        2. Flat dictionary format (e.g., {'source': '...', 'target': '...'})
        3. Custom formats via preprocessing_fn

        Args:
            example (dict): A single example from the dataset

        Returns:
            dict: Dictionary with 'source' and 'target' text
        """
        if self.config.preprocessing_fn is not None:
            return self.config.preprocessing_fn(example)

        source_col = self.config.source_column
        target_col = self.config.target_column or source_col

        if isinstance(example[source_col], dict):
            # Handle nested dictionary format (e.g., WMT datasets)
            return {
                "source": example[source_col][self.config.source_lang],
                "target": example[target_col][self.config.target_lang],
            }
        else:
            return {
                "source": example[source_col],
                "target": example[target_col],
            }

    def compute_metrics_A_and_B(self, EvalGeneration):
        return eval_cls.compute(predictions=EvalGeneration.predictions, references=EvalGeneration.labels)

    def compute_metrics(self) -> dict[str, Callable[[EvalGeneration], dict[str, float]]]:
        return {"A": self.compute_metrics_A_and_B, "B": self.compute_metrics_A_and_B}

    def preprocess(self, dataset: DatasetDict | IterableDatasetDict) -> tuple[DatasetDict, DatasetDict]:
        """Preprocess the dataset into two separate datasets for cycle training.

        Args:
            dataset (DatasetDict | IterableDatasetDict): The raw dataset containing translation pairs

        Returns:
            tuple[DatasetDict, DatasetDict]: Two datasets:
                - Dataset A: Source language texts
                - Dataset B: Target language texts
                Each containing 'train' and 'test' splits with parallel data in test
        """
        original_cols = set(dataset["train"].column_names)
        dataset = dataset.map(self._extract_text_pair)
        dataset = dataset.remove_columns(original_cols - {"source", "target"})

        dataset_A = dataset.map(lambda x: {"text": x["source"], "labels": x["target"]})
        dataset_B = dataset.map(lambda x: {"text": x["target"], "labels": x["source"]})

        dataset_A["train"] = dataset_A["train"].remove_columns(["labels"])
        dataset_B["train"] = dataset_B["train"].remove_columns(["labels"])

        return dataset_A, dataset_B
