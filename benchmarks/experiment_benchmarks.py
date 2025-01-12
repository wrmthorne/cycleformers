import argparse
import os
from pathlib import Path
import shutil
import gc
import torch
import yaml
from copy import deepcopy

import cycleformers
from cycleformers import CycleTrainer, CycleTrainingArguments
from cycleformers.utils import get_peft_config
from cycleformers.task_processors import AutoProcessor
from cycleformers.cycle_trainer_utils import load_model
from cycleformers.model_config import ModelConfig


os.environ["WANDB_PROJECT"] = "cycleformers-benchmarks"


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_training_args(global_config, dataset_config, mode_config):
    """Merge training arguments from different config levels."""
    training_args = {
        **(global_config.get('trainer_args', {})),
        **(dataset_config.get('trainer_args', {})),
        **(mode_config.get('trainer_args', {})),
    }
    
    # Convert numeric values
    for key in ['learning_rate', 'max_steps', 'num_train_epochs', 
                'per_device_train_batch_size', 'gradient_accumulation_steps']:
        if key in training_args:
            training_args[key] = float(training_args[key]) if key == 'learning_rate' else int(training_args[key])
    
    return training_args


def run_benchmark(training_args, model_config, dataset_A, dataset_B, compute_metrics, dataset_name, model_name, output_dir, use_macct,peft_config=None):
    print(f"\nRunning benchmark for {dataset_name} with {model_name} - MACCT mode: {use_macct}\n")
    print(f"\nTraining args: {training_args}\n")

    cycleformers_version = cycleformers.__version__
    tags = [
        dataset_name,
        f"v{cycleformers_version}",
        model_name,
        "macct" if use_macct else "dual-model"
    ]
    
    os.environ["WANDB_TAGS"] = ",".join(tags)

    training_args["output_dir"] = os.path.join(output_dir, f"{dataset_name}_{model_name}")
    training_args = CycleTrainingArguments(**training_args)
    
    # Load model(s)
    if use_macct:
        model_A = load_model(model_config.model_name_or_path, **{"config": model_config, **training_args.model_init_kwargs})
        model_B = None
        models = model_A
    else:
        model_A = load_model(model_config.model_name_or_path, **{"config": model_config, **training_args.model_init_kwargs})
        model_B = load_model(model_config.model_name_or_path, **{"config": model_config, **training_args.model_init_kwargs})
        models = {"A": model_A, "B": model_B}
    
    trainer = CycleTrainer(
        training_args,
        models=models,
        train_dataset_A=dataset_A['train'],
        train_dataset_B=dataset_B['train'],
        eval_dataset_A=dataset_A['test'] if training_args.eval_strategy != "no" else None,
        eval_dataset_B=dataset_B['test'] if training_args.eval_strategy != "no" else None,
        peft_configs=peft_config,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    trainer.evaluate()
    
    # Clean up
    del trainer, models, model_A, model_B
    torch.cuda.empty_cache()
    gc.collect()
    
    # Remove cache directories
    cache_dir = os.path.join(training_args.output_dir, "cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

def get_training_config(config, dataset_name, model_name):
    """Get configuration for training from the new config structure.
    
    Args:
        config: The loaded config dictionary
        dataset_name: Name of the dataset to use
        model_name: Name of the model to use
        use_macct: Whether to use MACCT mode
    """
    # Get dataset config
    if dataset_name not in config['datasets']:
        raise ValueError(f"Dataset {dataset_name} not found in config")
    dataset_config = config['datasets'][dataset_name]['dataset_config']
    
    # Get model config
    if model_name not in config['models']:
        raise ValueError(f"Model {model_name} not found in config")
    model_config = config['models'][model_name]
    
    # Get base model args
    model_args = model_config['model_args'].copy()

    # Get training args from all levels and merge
    training_args = config.get('training_args', {}).copy()
    use_macct = training_args.get('use_macct', False)
    
    # Get mode-specific config and merge with base args
    mode_config = model_config['macct' if use_macct else 'dual_model']
    if 'model_args' in mode_config:
        model_args.update(mode_config['model_args'])

    if 'trainer_args' in mode_config:
        training_args.update(mode_config['trainer_args'])
    
    # Create model config object
    base_model_config = ModelConfig(**model_args)
    
    # Get training args from all levels and merge
    training_args = config.get('training_args', {}).copy()
    if 'trainer_args' in mode_config:
        training_args.update(mode_config['trainer_args'])
    
    # Only get peft config if in macct mode and peft is enabled
    peft_config = None
    if use_macct and model_args.get('use_peft', False):
        peft_config = get_peft_config(base_model_config)
    
    return dataset_config, training_args, base_model_config, peft_config, use_macct

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, nargs="+", default=None, 
                        help="Tasks to benchmark. If not provided, all tasks will be benchmarked.")
    parser.add_argument("--output_dir", type=str, default="benchmark_models/",
                        help="Directory to save model outputs")
    parser.add_argument("--config", type=str, default=Path(__file__).parent / "configs/exp_config.yaml",
                        help="Path to experiment config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    tasks = args.tasks if args.tasks else list(config['datasets'].keys())

    for task in tasks:
        if task not in config['datasets']:
            print(f"Warning: Task {task} not found in benchmark configs. Skipping.")
            continue
        
        for model_name in config['models']:
            
            dataset_config, training_args, model_config, peft_config, use_macct = get_training_config(
                config, task, model_name
            )
            
            processor = AutoProcessor.load_processor(task, **dataset_config)
            dataset_A, dataset_B = processor.process()
            
            run_benchmark(
                training_args=training_args,
                model_config=model_config,
                dataset_A=dataset_A,
                dataset_B=dataset_B,
                compute_metrics=processor.compute_metrics,
                dataset_name=task,
                model_name=model_name,
                output_dir=args.output_dir,
                use_macct=use_macct,
                peft_config=peft_config
            )

if __name__ == "__main__":
    main()
    


    