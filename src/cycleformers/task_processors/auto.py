import importlib
from dataclasses import dataclass
from typing import TypeVar

from .base import BaseProcessor, ProcessorConfig


T = TypeVar("T", bound="BaseProcessor")


@dataclass
class ProcessorMapping:
    processor: str
    config: str


PROCESSOR_MAPPING = {
    "conll2003": ProcessorMapping(
        "cycleformers.task_processors.ner.CONLL2003Processor",
        "cycleformers.task_processors.ner.CONLL2003ProcessorConfig",
    ),
    "wmt14": ProcessorMapping(
        "cycleformers.task_processors.translation.TranslationProcessor",
        "cycleformers.task_processors.translation.TranslationProcessorConfig",
    ),
}


class AutoProcessor:
    """
    Factory class for automatically selecting and configuring NLP dataset processors.

    Maps dataset names to their corresponding processor classes for NLP tasks like
    named entity recognition, machine translation, text classification, etc.

    Example:
        >>> config = TranslationProcessorConfig(dataset_name="wmt14", eval_split_ratio=0.2)
        >>> processor = AutoProcessor.from_config(config)
        >>> dataset_A, dataset_B = processor.process()
    """

    @classmethod
    def load_processor(cls, task: str, **config_kwargs) -> BaseProcessor:
        config_cls = cls.get_config_class(task)
        config = config_cls(**config_kwargs)
        processor_cls = cls.get_processor_class(task)
        return processor_cls(config)

    @staticmethod
    def _import_task_class(class_path: str) -> type[BaseProcessor] | type[ProcessorConfig]:
        """Dynamically import a processor class from its string path."""
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import class '{class_path}': {str(e)}")

    @classmethod
    def get_processor_class(cls, dataset_name: str) -> type[BaseProcessor]:
        """Get the appropriate processor class for a given dataset."""
        dataset_name = str(dataset_name).lower().strip()
        if dataset_name in PROCESSOR_MAPPING:
            processor_class = cls._import_task_class(PROCESSOR_MAPPING[dataset_name].processor)
            assert issubclass(processor_class, BaseProcessor)
            return processor_class

        raise ValueError(
            f"No processor found for dataset: {dataset_name}. "
            f"Available datasets: {', '.join(sorted(PROCESSOR_MAPPING.keys()))}"
        )

    @classmethod
    def get_config_class(cls, dataset_name: str) -> type[ProcessorConfig]:
        """Get the appropriate processor class for a given dataset."""
        dataset_name = str(dataset_name).lower().strip()
        if dataset_name in PROCESSOR_MAPPING:
            config_class = cls._import_task_class(PROCESSOR_MAPPING[dataset_name].config)
            assert issubclass(config_class, ProcessorConfig)
            return config_class

        raise ValueError(
            f"No config found for dataset: {dataset_name}. "
            f"Available datasets: {', '.join(sorted(PROCESSOR_MAPPING.keys()))}"
        )

    @classmethod
    def from_config(cls, config: ProcessorConfig) -> BaseProcessor:
        """Create and configure a processor instance from a configuration object."""
        if config.dataset_name is None:
            raise ValueError("dataset_name must be provided in config")

        processor_cls = cls.get_processor_class(str(config.dataset_name))
        return processor_cls(config)
