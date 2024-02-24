import os
from typing import Callable, Dict, List, Optional, Tuple, Union

from datasets import Dataset
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    DataCollator,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    default_data_collator,
    EvalPrediction,
    PreTrainedTokenizerBase,
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    TrainerState,
)
from transformers.optimization import get_scheduler
from transformers.trainer_utils import (
    enable_full_determinism,
    set_seed,
    TrainerMemoryTracker,
)
from transformers.utils import logging
from transformers.utils.import_utils import is_peft_available

from cycleformers.cycles.cycle_utils import init_cycle
from cycleformers.data import CombinedLoader
from .training_args import ModelTrainingArguments, TrainingArguments
from .model_handler import _ModelHandler
from .trainer_utils import (
    get_parameter_names,
    has_length,
    validate_collator_new,
    validate_tokenizer,
)

if is_peft_available():
    from peft import PeftModel


logger = logging.get_logger(__name__)


class CycleTrainer(Trainer):
    '''
    CycleTrainer is a wrapper class for the Huggingface Trainer class, specifically designed to
    train models using cycle consistency training but aiming to be a generic multi-model trainer.

    Args:
        models (`dict` or `PeftModel`, *required*):
            The models to be trained. If a dictionary is supplied, the keys should be the names of the
            models and the values should be the models themselves. If a PeftModel is supplied, there must
            be at least two attached adapters and the model will be trained using those adapters.
        tokenizers (`dict` or `PreTrainedTokenizerBase`, *required*):
            The tokenizers to be used for each model. If a dictionary is supplied, the keys should be the
            names of the models and the values should be the tokenizers themselves. If a single tokenizer
            is supplied, it will be used for all models.
        args (`TrainerConfig`, *optional*):
            General, non model-specific training arguments. If not supplied, default arguments will be used.
        model_args (`dict`, *optional*):
            The model arguments to be used for training each model. The keys should be the names of the models
            and the values should be the model arguments themselves. If a single set of model arguments is
            supplied, it will be used for all models. If no model arguments are supplied for a specific model,
            the default model arguments will be used.
        data_collators (`dict` or `DataCollator`, *optional*):
            The function to use to form a batch from a list of elements of the dataset. If a dictionary is
            supplied, the keys should be the names of the models and the values should be the data collators
            themselves. If a single data collator is supplied, it will be used for all models. If no collator
            is supplied for a specific model, a collator will be chosen based on the model type.
        train_datasets (`dict`, *optional*):
            The training datasets to be used for each model. The keys should be the names of the models and
            the values should be the datasets themselves.
        eval_datasets (`dict`, *optional*):
            The evaluation datasets to be used for each model. The keys should be the names of the models and
            the values should be the datasets themselves. A subset of models can be evaluated by supplying
            only a subset of model names.
        optimizers (`dict`, *optional*):
            The optimizers to be used for each model. The keys should be the names of the models and the
            values should be a tuple of the optimizer and the learning rate scheduler. Will default to
            AdamW and a linear learning rate scheduler if nothing is supplied for any specific model.
    '''
    # TODO: Fix type hinting
    def __init__(
        self,
        models: Union[Dict[str, Union[PreTrainedModel, nn.Module]], 'PeftModel'],
        tokenizers: Union[Dict[str, PreTrainedTokenizerBase], PreTrainedTokenizerBase],
        args: Optional[TrainingArguments] = None,
        model_args: Optional[Union[Dict[str, ModelTrainingArguments], ModelTrainingArguments]] = dict(),
        data_collators: Optional[Union[Dict[str, DataCollator], DataCollator]] = dict(),
        train_datasets: Optional[Dict[str, Dataset]] = dict(),
        eval_datasets: Optional[Dict[str, Dataset]] = None,
        model_init: Optional[Dict[str, Callable[[], PreTrainedModel]]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[Union[Dict[str, List[TrainerCallback]], List[TrainerCallback]]] = None,
        optimizers: Optional[Dict[str, Tuple[Optimizer, LambdaLR]]] = dict(),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        # TODO: Load Trainer args as default and overwrite them with model args if present. Warn if overwriting
        if args is None:
            output_dir = 'tmp_trainer'
            logger.info(f'No `TrainingArguments` passed, using `output_dir={output_dir}`.')
            args = TrainingArguments(output_dir=output_dir)
        self.args = args
        # Seed must be set before instantiating the models when using models
        enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
        self.is_in_train = False

        self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        self._memory_tracker.start()

        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)

        # TODO: Handle case where any arg is not a dict
        self._model_names = list(models.keys())

        self.handlers = {
            name: _ModelHandler(
                models[name],
                model_args.get(name, ModelTrainingArguments(
                    output_dir=os.path.join(args.output_dir, name)
                ).update_from_global_args(args)),
                tokenizers[name],
                model_init=model_init.get(name, None),
                compute_metrics=compute_metrics.get(name, None),
                callbacks=callbacks.get(name, None),
                optimizers=optimizers.get(name, (None, None)),
                preprocess_logits_for_metrics=preprocess_logits_for_metrics.get(name, None),
            ) for name in self.model_names
        }

        for name, handler in self.handlers.items():
            collator = data_collators.get(name, None)
            if collator is None:
                if handler.tokenizer is None:
                    data_collators[name] = default_data_collator
                else:
                    data_collators[name] = DataCollatorWithPadding(handler.tokenizer)
            else:
                if not callable(collator) and callable(getattr(collator, "collate_batch", None)):
                    raise ValueError(
                        f'Invalid data collator for model {name}. Must be a callable and have a '
                         'collate_batch method.'
                    )

        if not set(self._model_names) == set(train_datasets.keys()):
            raise ValueError('Training datasets must be supplied for all models.')
        
        if any(name not in self._model_names for name in eval_datasets):
            raise ValueError(
                f'Unrecognised model names in eval_datasets. Expected {self._model_names}, '
                f'got {list(eval_datasets.keys())}.'
            )
        
        self.data_collators = data_collators
        self.train_datasets = train_datasets
        self.eval_datasets = eval_datasets

        # TODO: FINISH THIS
        

        self._memory_tracker.stop_and_update_metrics()
            

    def get_train_dataloader(self) -> DataLoader:
        '''
        Method to create the training dataloader for the models.
        
        Returns:
            CombinedLoader: The combined dataloader for the all models.
        '''
        if self.train_datasets is None:
            raise ValueError('No training datasets found. Please supply training datasets.')
        
        data_loaders = {}
        for name, dataset in self.train_datasets.items():
            data_loaders[name] = DataLoader(
                dataset,
                batch_size=self.model_args[name].per_device_train_batch_size,
                shuffle=True,
                collate_fn=self.data_collators[name]
            )

        return CombinedLoader(data_loaders, 'max_size')
    

    def get_eval_dataloader(self, eval_datasets: Optional[Dict[str, Dataset]] = None) -> DataLoader:
        '''
        Method to create the evaluation dataloader for the models.
        
        Args:
            eval_datasets (`dict`, *optional*):
                The evaluation datasets to be used for each model. The keys should be the names of the models and
                the values should be the datasets themselves. A subset of models can be evaluated by supplying
                only a subset of model names.
                
        Returns:
            CombinedLoader: The combined dataloader for the all models.
        '''
        if (len(eval_datasets) == 0 or eval_datasets is None) and \
           (len(self.eval_datasets) == 0 or self.eval_datasets is None):
            raise ValueError('No evaluation datasets found. Please supply evaluation datasets.')
        
        eval_datasets = eval_datasets if eval_datasets is not None else self.eval_datasets

        data_loaders = {}
        for name, dataset in eval_datasets.items():
            data_loaders[name] = DataLoader(
                dataset,
                batch_size=self.model_args[name].per_device_eval_batch_size,
                shuffle=True,
                collate_fn=self.data_collators[name]
            )

        return CombinedLoader(data_loaders, 'max_size')
    

    def create_optimizers_and_schedulers(self):
        '''
        Method to create the optimizers and learning rate schedulers for the models.
        '''
        if any(name not in self.model_names for name in self.optimizers.keys()):
            raise ValueError(
                f'Optimizers contains a key not found in models. Expected keys to be a subset of '
                f'{self.model_names}, got {list(self.optimizers.keys())}.'
            )

        self.lr_schedulers = {}
        for name, model in self.models.items():
            if self.optimizers.get(name, (None, None)) == (None, None):
                decay_parameters = self.get_decay_parameter_names(model)
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.model_args[name].weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.model_args[name])
                optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                scheduler = get_scheduler(
                    self.model_args[name].lr_scheduler_type,
                    optimizer=optimizer, 
                    num_warmup_steps=self.model_args[name].get_warmup_steps(-1),
                    num_training_steps=self.model_args[name].max_steps,
                    scheduler_specific_kwargs=self.model_args[name].lr_scheduler_kwargs,
                )

                self.optimizers[name] = optimizer
                self.lr_schedulers[name] = scheduler

    def train(self):
        '''
        Main training entry point.

        MORE COMPATABILITY STUFF WILL BE ADDED HERE EVENTUALLY
        '''
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        
        args = self.args

        self.is_in_train = True

        return self._inner_training_loop()
    
    def _inner_training_loop(self):
        '''
        Inner training loop for the CycleTrainer.

        MORE COMPATABILITY STUFF WILL BE ADDED HERE EVENTUALLY
        '''
        
        train_dataloader = iter(self.get_train_dataloader()) # TODO: Fix this stupid need to call iter

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            num_examples = self.num_examples(train_dataloader)
            

        self.global_state = TrainerState() # TODO: How to handle state for multiple sub models?

        self.model_states = {}
        for name, model in self.models.items():
            self.model_states[name] = TrainerState()
            model.zero_grad()

        # Train!
        logger.info('***** Running training *****')
        logger.info(f'  Num examples = {num_examples:,}')


        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        total_batched_samples = 0
        # TODO: Handle variable number of epochs between models
        for epoch in range(int(max(self.model_args[name].num_train_epochs for name in self.model_names))):
            epoch_iterator = train_dataloader

            steps_in_epoch = len(epoch_iterator)

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1

                for name in self.model_names:
                    model = self.models[name]
                    model_inputs = inputs[name]

                    if model_inputs is None:
                        continue

                    tr_loss_step = self.training_step(self.cycles[name], model_inputs) / self.model_args[name].gradient_accumulation_steps
                    print(tr_loss_step)

                    if total_batched_samples % self.model_args[name].gradient_accumulation_steps == 0:
                        self.optimizers[name].step()
                        self.lr_schedulers[name].step()

                    model.zero_grad()

        
        metrics = {}
        self._memory_tracker.stop_and_update_metrics(metrics)


    def training_step(self, model, inputs):
        model = model

        loss = self.compute_loss(model, inputs)
        loss.backward()

        return loss.detach()



    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    

    def log(self, logs):
        if self.global_state.epoch is not None:
            logs['epoch'] = round(self.global_state.epoch, 2)
        if self.args.include_num_input_tokens_seen:
            logs['num_input_tokens_seen'] = self.global_state.num_input_tokens

        output = {**logs, **{"step": self.global_state.global_step}}

        