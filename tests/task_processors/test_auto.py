import pytest

from cycleformers.task_processors import (
    CONLL2003Processor,
    CONLL2003ProcessorConfig,
    TranslationProcessor,
    TranslationProcessorConfig,
)
from cycleformers.task_processors.auto import AutoProcessor, ProcessorConfig


@pytest.mark.parametrize(
    "dataset_name, config_cls, expected_processor",
    [
        ("conll2003", CONLL2003ProcessorConfig, CONLL2003Processor),
        ("wmt14", TranslationProcessorConfig, TranslationProcessor),
    ],
)
class TestAutoProcessor:
    """Tests for the AutoProcessor class."""

    def test_get_processor_class_valid_dataset(self, dataset_name, config_cls, expected_processor):
        """Test getting processor class for a valid dataset name."""
        processor_cls = AutoProcessor.get_processor_class(dataset_name)
        assert processor_cls == expected_processor

    def test_get_processor_class_invalid_dataset(self, dataset_name, config_cls, expected_processor):
        """Test that appropriate error is raised for invalid dataset names."""
        with pytest.raises(ValueError, match="No processor found for dataset"):
            AutoProcessor.get_processor_class("invalid_dataset")

    def test_from_config_valid(self, dataset_name, config_cls, expected_processor):
        """Test creating processor from valid config."""
        config = config_cls(dataset_name=dataset_name, eval_split_ratio=0.2)
        processor = AutoProcessor.from_config(config)
        assert isinstance(processor, expected_processor)
        assert processor.config == config

    def test_from_config_missing_dataset(self, dataset_name, config_cls, expected_processor):
        """Test that appropriate error is raised when dataset_name is missing."""
        config = ProcessorConfig(dataset_name=None)
        with pytest.raises(ValueError, match="dataset_name must be provided in config"):
            AutoProcessor.from_config(config)

    def test_import_processor_class_invalid_path(self, dataset_name, config_cls, expected_processor):
        """Test that appropriate error is raised for invalid import paths."""
        with pytest.raises(ImportError):
            AutoProcessor._import_processor_class("invalid.module.path")

    def test_import_processor_class_invalid_class(self, dataset_name, config_cls, expected_processor):
        """Test that appropriate error is raised for valid module but invalid class."""
        with pytest.raises(ImportError):
            AutoProcessor._import_processor_class("cycleformers.task_processors.ner.InvalidClass")
