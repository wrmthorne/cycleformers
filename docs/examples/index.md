# Examples

This section contains practical examples of using Cycleformers for various tasks. While more detailed examples are under construction, here are some key examples to get you started:

## Machine Translation

### WMT-14 English-German Translation
[View Example](wmt_2014.md)

A basic example showing how to train a translation model using the WMT-14 English-German dataset. This example demonstrates:
- Loading and preprocessing WMT-14 data
- Setting up a basic MACCT configuration
- Training with memory-efficient adapters

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from cycleformers import CycleTrainer, CycleTrainingArguments
from datasets import load_dataset

# Load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")

# Load WMT14 dataset
dataset = load_dataset("wmt14", "de-en")
train_en = dataset["train"].select_columns(["translation"]).map(lambda x: {"text": x["translation"]["en"]})
train_de = dataset["train"].select_columns(["translation"]).map(lambda x: {"text": x["translation"]["de"]})

# Basic training setup
args = CycleTrainingArguments(
    output_dir="wmt14-translation",
    per_device_train_batch_size=8,
    num_train_epochs=3
)

trainer = CycleTrainer(
    args=args,
    models=model,
    tokenizers=tokenizer,
    train_dataset_A=train_en,
    train_dataset_B=train_de
)

trainer.train()
```

🚧 More detailed examples coming soon, including:
- Advanced configuration options
- Evaluation metrics
- Model checkpointing

## Named Entity Recognition

### Cross-lingual NER
[View Example](cycle_ner.md)

Example of using cycle-consistency training for cross-lingual named entity recognition:
- Transfer learning between languages
- Using custom task processors
- Handling structured prediction tasks

🚧 Under construction:
- Detailed data preprocessing steps
- Model evaluation code
- Performance benchmarks

## Coming Soon

We're working on additional examples including:
- Style transfer
- Text simplification
- Cross-lingual summarization
- Custom task processors

Want to contribute an example? Check our [Contributing Guide](../contributing.md)!