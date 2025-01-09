# Task Processors

🚧 This section is under construction. 🚧

When using task_processors, which splits are downloaded and the size of them can be controlled via the datasets `slice split` syntax. For example, the following will download the first 100 samples from the train split of the WMT14 dataset and the first 30 samples from the test split.

```python
from datasets import load_dataset
from cycleformers.task_processors.translation import TranslationProcessor

config = TranslationProcessorConfig(
    dataset_name="wmt14",
    dataset_config_name="de-en",
    split=["train[:100]", "test[:30]"],
)
dataset = TranslationProcessor(config)
```

More information on the syntax for `slice split` can be found in the [datasets documentation](https://huggingface.co/docs/datasets/loading#slice-splits).

## BaseProcessor

::: src.cycleformers.task_processors.base.BaseProcessor

::: src.cycleformers.task_processors.base.ProcessorConfig


## Named-Entity Recognition (NER)

::: src.cycleformers.task_processors.ner.CONLL2003Processor

::: src.cycleformers.task_processors.ner.CONLL2003ProcessorConfig


## Translation (e.g. WMT)

::: src.cycleformers.task_processors.translation.TranslationProcessor

::: src.cycleformers.task_processors.translation.TranslationProcessorConfig


### Helper Functions

::: src.cycleformers.task_processors.ner.reconstruct_sentence

::: src.cycleformers.task_processors.ner.ner_to_sequences
