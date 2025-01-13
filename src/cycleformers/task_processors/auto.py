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
    """Factory class for automatically selecting and configuring task-specific dataset processors.

    This class provides a simple interface for loading the appropriate processor for a given dataset.
    It handles the mapping between dataset names and their corresponding processor classes.

    Currently supported datasets:
        - `"conll2003"`: Named Entity Recognition using the CONLL2003 format
        - `"wmt14"`: Machine Translation using the WMT14 format

    Example:
        >>> from cycleformers.task_processors import AutoProcessor
        >>> from cycleformers.task_processors.translation import TranslationProcessorConfig
        >>>
        >>> # Load a processor for WMT14 translation
        >>> config = TranslationProcessorConfig(
        ...     dataset_name="wmt14",
        ...     dataset_config_name="de-en",
        ...     source_lang="en",
        ...     target_lang="de"
        ... )
        >>> processor = AutoProcessor.from_config(config)
        >>> dataset_A, dataset_B = processor.process()
        >>>
        >>> # Load a processor directly with kwargs
        >>> processor = AutoProcessor.load_processor(
        ...     "wmt14",
        ...     dataset_name="wmt14",
        ...     dataset_config_name="de-en",
        ...     source_lang="en",
        ...     target_lang="de"
        ... )
        >>> dataset_A, dataset_B = processor.process()

    For more details on task processors and their configurations, see the
    [documentation](https://wrmthorne.github.io/cycleformers/conceptual_reference/task_processors).
    """

    @classmethod
    def load_processor(cls, task: str, **config_kwargs) -> BaseProcessor:
        """Load a processor for a given task with configuration parameters.

        This is a convenience method that creates a configuration object from kwargs
        and instantiates the appropriate processor.

        Args:
            task (str): The name of the task/dataset (e.g., "wmt14", "conll2003")
            **config_kwargs: Configuration parameters specific to the task

        Returns:
            BaseProcessor: An instance of the appropriate processor for the task

        Example:
            >>> processor = AutoProcessor.load_processor(
            ...     "wmt14",
            ...     dataset_name="wmt14",
            ...     dataset_config_name="de-en",
            ...     source_lang="en",
            ...     target_lang="de"
            ... )
        """
        config_cls = cls.get_config_class(task)
        config = config_cls(**config_kwargs)
        processor_cls = cls.get_processor_class(task)
        return processor_cls(config)

    @staticmethod
    def _import_task_class(class_path: str) -> type[BaseProcessor] | type[ProcessorConfig]:
        """Dynamically import a processor or config class from its string path.

        Args:
            class_path (str): Fully qualified path to the class (e.g., "cycleformers.task_processors.ner.CONLL2003Processor")

        Returns:
            type[BaseProcessor] | type[ProcessorConfig]: The imported class

        Raises:
            ImportError: If the class cannot be imported
        """
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import class '{class_path}': {str(e)}")

    @classmethod
    def get_processor_class(cls, dataset_name: str) -> type[BaseProcessor]:
        """Get the appropriate processor class for a given dataset.

        Args:
            dataset_name (str): Name of the dataset (e.g., "wmt14", "conll2003")

        Returns:
            type[BaseProcessor]: The processor class for the dataset

        Raises:
            ValueError: If no processor is found for the dataset
        """
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
        """Get the appropriate configuration class for a given dataset.

        Args:
            dataset_name (str): Name of the dataset (e.g., "wmt14", "conll2003")

        Returns:
            type[ProcessorConfig]: The configuration class for the dataset

        Raises:
            ValueError: If no configuration is found for the dataset
        """
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
        """Create and configure a processor instance from a configuration object.

        Args:
            config (ProcessorConfig): Configuration object for the processor

        Returns:
            BaseProcessor: An instance of the appropriate processor

        Raises:
            ValueError: If dataset_name is not provided in the config
        """
        if config.dataset_name is None:
            raise ValueError("dataset_name must be provided in config")

        processor_cls = cls.get_processor_class(str(config.dataset_name))
        return processor_cls(config)
