from transformers import Trainer


# TODO: Needs major reworking to support useful functionality, edge cases and later flexibility
class CycleTrainer(Trainer):
    def __init__(
        self,
        model=None,
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
        preprocess_logits_for_metrics=None
    ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, processing_class, model_init, compute_loss_func, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)

    def evaluate(self, eval_dataset, **kwargs):
        raise NotImplementedError("Evaluation is not supported yet")