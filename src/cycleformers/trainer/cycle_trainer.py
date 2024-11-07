import sys

from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.trainer_utils import seed_worker


if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from cycleformers.data.samplers import CyclicSampler


# TODO: Needs major reworking to support useful functionality, edge cases and later flexibility
class CycleTrainer(Trainer):
    def __init__(
        self,
        model: PeftModel = None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        model_init=None,
        compute_loss_func=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
    ):
        # Hack solution - needs replacing. Multiply gradient accumulation by number of models to cause all models to be updated on one step to not have their gradients zeroed out
        args.gradient_accumulation_steps *= len(list(model.peft_config))

        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            processing_class,
            model_init,
            compute_loss_func,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

    # TODO: Investigate potential better ways to handle this
    @override
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`] with interleaved batch sampling strategy.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # TODO: Create batch sampler with same configs as the dataset
        sampler = CyclicSampler(
            *self.train_dataset,
            batch_size=self._train_batch_size,
            # stop_on=self.args.stop_on, # TODO: Add this back in with CycleConfig later
        )

        dataloader_params = {
            "batch_size": None,  # Batching is handled by sampler
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "drop_last": self.args.dataloader_drop_last,  # May need to be moved to sampler
            "worker_init_fn": seed_worker,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
        }
        return self.accelerator.prepare(DataLoader(sampler, **dataloader_params))

    @override
    def evaluate(self, eval_dataset, **kwargs):
        raise NotImplementedError("Evaluation is not supported yet")
