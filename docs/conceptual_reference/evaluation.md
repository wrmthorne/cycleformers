# Evaluation in Cycleformers

Cycleformers provides a comprehensive evaluation framework that integrates with task-specific metrics and supports both standard dual-model and memory-efficient adapter-based training.

## Dataset Structure

### Training vs Test Datasets

The key difference between training and test datasets lies in their structure:

1. **Training Datasets**:
   - Contains only input text in the `text` column
   - No labels/targets are provided
   - Used for generating synthetic data during cycle training
   ```python
   train_dataset = {
       "text": ["The cat sat on the mat", "John works at Google"]
   }
   ```

2. **Test Datasets**:
   - Contains both input text and reference labels
   - Labels are used to compute evaluation metrics
   - Maintains parallel data for accurate evaluation
   ```python
   test_dataset = {
       "text": ["The cat sat on the mat", "John works at Google"],
       "labels": ["Die Katze saß auf der Matte", "John | person Google | organization"]
   }
   ```

## Task-Specific Metrics

Each task processor defines its own evaluation metrics through the `compute_metrics()` method. The method returns either:
- A single callable for both models
- A dictionary mapping model names to their respective metric functions

### Translation Metrics

The `TranslationProcessor` uses a combination of standard MT metrics:
```python
metrics = {
    "sacrebleu": evaluate.load("sacrebleu"),
    "rouge": evaluate.load("rouge"),
}
```

### NER Metrics

The `CONLL2003Processor` uses the seqeval metric for entity-level evaluation:
```python
metrics = evaluate.load("seqeval")
```

## Integration with Task Processors

To implement custom metrics for a new task:

1. Create your metric computation function:
```python
def compute_task_metrics(eval_pred: EvalGeneration) -> dict[str, float]:
    predictions = eval_pred.predictions  # List of generated texts
    references = eval_pred.labels       # List of reference texts
    
    # Compute your metrics
    results = calculate_metrics(predictions, references)
    return results
```

2. Implement the `compute_metrics()` method in your processor:
```python
def compute_metrics(self) -> dict[str, Callable[[EvalGeneration], dict[str, float]]]:
    # Same metrics for both models
    return {
        "A": self.compute_task_metrics,
        "B": self.compute_task_metrics
    }
    
    # Or different metrics per model
    return {
        "A": self.compute_metrics_A,
        "B": self.compute_metrics_B
    }
```

## Evaluation Process

The evaluation process in cycleformers follows these steps:

1. **Model Generation**:
   ```python
   outputs = model.generate(
       input_ids=batch["input_ids"],
       attention_mask=batch["attention_mask"]
   )
   ```

2. **Text Decoding**:
   ```python
   predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
   labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
   ```

3. **Metric Computation**:
   ```python
   eval_preds = EvalGeneration(predictions=predictions, labels=labels)
   metrics = compute_metrics_fn(eval_preds)
   ```

4. **Metric Logging**:
   - Metrics are prefixed with model name: `eval_metric_A`, `eval_metric_B`
   - Logged to whatever tracking system is configured (e.g., wandb, tensorboard)

## Customizing Evaluation

### Custom Metrics

You can provide custom metrics when initializing the trainer:
```python
def custom_metrics(eval_pred: EvalGeneration) -> dict[str, float]:
    return {
        "my_metric": compute_my_metric(eval_pred.predictions, eval_pred.labels)
    }

trainer = CycleTrainer(
    # ... other args ...
    compute_metrics=custom_metrics,  # Same for both models
    # Or
    compute_metrics={"A": metrics_a, "B": metrics_b}  # Different per model
)
```

### Evaluation Frequency

Control when evaluation happens through training arguments:
```python
args = CycleTrainingArguments(
    eval_steps=100,           # Evaluate every 100 steps
    eval_on_start=True,       # Evaluate before training
    evaluation_strategy="steps"  # Or "epoch" or "no"
)
```

### Manual Evaluation

You can also trigger evaluation manually:
```python
# Evaluate on default test sets
metrics = trainer.evaluate()

# Evaluate on custom datasets
metrics = trainer.evaluate({
    "A": custom_dataset_a,
    "B": custom_dataset_b
})
```
