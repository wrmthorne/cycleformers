# Task Processors

Task processors handle the conversion of various dataset formats into cycleformers-compatible format. They provide a flexible API for preparing datasets for cycle training.

## AutoProcessor

::: cycleformers.task_processors.AutoProcessor

The `AutoProcessor` is a factory class that automatically selects and configures the appropriate processor for a given dataset. It provides a simple interface for loading task-specific processors without needing to know their implementation details.

### Key Features

- Automatic processor selection based on dataset name
- Support for custom processor configurations
- Easy-to-use interface for loading and configuring processors
- Extensible mapping system for adding new processors

### Methods

#### load_processor

::: cycleformers.task_processors.auto.AutoProcessor.load_processor

#### from_config

::: cycleformers.task_processors.auto.AutoProcessor.from_config

#### get_processor_class

::: cycleformers.task_processors.auto.AutoProcessor.get_processor_class

#### get_config_class

::: cycleformers.task_processors.auto.AutoProcessor.get_config_class

### Example Usage

```python
from cycleformers.task_processors import AutoProcessor
from cycleformers.task_processors.translation import TranslationProcessorConfig

# Method 1: Load with configuration object
config = TranslationProcessorConfig(
    dataset_name="wmt14",
    dataset_config_name="de-en",
    source_lang="en",
    target_lang="de"
)
processor = AutoProcessor.from_config(config)
dataset_A, dataset_B = processor.process()

# Method 2: Load directly with kwargs
processor = AutoProcessor.load_processor(
    "wmt14",
    dataset_name="wmt14",
    dataset_config_name="de-en",
    source_lang="en",
    target_lang="de"
)
dataset_A, dataset_B = processor.process()
```

### Supported Datasets

Currently supported datasets and their corresponding processors:

| Dataset | Task | Processor | Configuration |
|---------|------|-----------|---------------|
| `"wmt14"` | Machine Translation | `TranslationProcessor` | `TranslationProcessorConfig` |
| `"conll2003"` | Named Entity Recognition | `CONLL2003Processor` | `CONLL2003ProcessorConfig` |

### Adding New Processors

To add support for a new dataset:

1. Create your processor and configuration classes
2. Add an entry to the `PROCESSOR_MAPPING` in `AutoProcessor`:

```python
PROCESSOR_MAPPING = {
    "your_dataset": ProcessorMapping(
        "path.to.your.CustomProcessor",
        "path.to.your.CustomProcessorConfig",
    ),
}
```

## BaseProcessor

::: cycleformers.task_processors.BaseProcessor

The `BaseProcessor` is the abstract base class that defines the interface for all task processors. It handles common functionality like dataset loading and train/test splitting.

### Key Methods

- `load()`: Load the source dataset
- `process()`: Process the dataset into two separate datasets A and B
- `preprocess()`: Task-specific preprocessing (must be implemented by subclasses)
- `compute_metrics()`: Task-specific evaluation metrics (must be implemented by subclasses)

## Available Processors

### TranslationProcessor

::: cycleformers.task_processors.TranslationProcessor

The `TranslationProcessor` handles machine translation datasets, supporting both standard parallel corpora and back-translation style training.

#### Configuration

::: cycleformers.task_processors.TranslationProcessorConfig

#### Example Usage

```python
from cycleformers.task_processors import TranslationProcessor
from cycleformers.task_processors.translation import TranslationProcessorConfig

config = TranslationProcessorConfig(
    dataset_name="wmt14",
    dataset_config_name="de-en",
    source_lang="en",
    target_lang="de"
)
processor = TranslationProcessor(config)
dataset_A, dataset_B = processor.process()

print(dataset_A["train"][0])  # {'text': 'The cat sat on the mat.'}
print(dataset_B["train"][0])  # {'text': 'Die Katze saß auf der Matte.'}
```

### CONLL2003Processor

::: cycleformers.task_processors.CONLL2003Processor

The `CONLL2003Processor` handles Named Entity Recognition (NER) datasets, converting between raw text and entity sequence formats.

#### Configuration

::: cycleformers.task_processors.CONLL2003ProcessorConfig

#### Example Usage

```python
from cycleformers.task_processors import CONLL2003Processor
from cycleformers.task_processors.ner import CONLL2003ProcessorConfig

config = CONLL2003ProcessorConfig(sep_token=" | ")
processor = CONLL2003Processor(config)
dataset_A, dataset_B = processor.process()

print(dataset_A["train"][0])  # {'text': 'John Smith works at Google.'}
print(dataset_B["train"][0])  # {'text': 'John Smith | person Google | organization'}
```

## Creating Custom Processors

To create a custom processor for a new task or dataset format:

1. Create a configuration class inheriting from `ProcessorConfig`
2. Create a processor class inheriting from `BaseProcessor`
3. Implement the required methods: `preprocess()` and `compute_metrics()`
4. (Optional) Add your processor to the `PROCESSOR_MAPPING` in `AutoProcessor`

Example:

```python
from cycleformers.task_processors import BaseProcessor, ProcessorConfig
from dataclasses import dataclass

@dataclass
class CustomProcessorConfig(ProcessorConfig):
    task_specific_param: str = "default"

class CustomProcessor(BaseProcessor):
    def __init__(self, config: CustomProcessorConfig = CustomProcessorConfig()):
        super().__init__(config)
        self.config: CustomProcessorConfig = config

    def preprocess(self, dataset):
        # Convert dataset into cycleformers format
        dataset_A = dataset.map(lambda x: {"text": process_for_A(x)})
        dataset_B = dataset.map(lambda x: {"text": process_for_B(x)})
        return dataset_A, dataset_B

    def compute_metrics(self):
        return {"A": compute_metrics_A, "B": compute_metrics_B}
```