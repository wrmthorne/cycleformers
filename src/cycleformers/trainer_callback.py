import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from transformers.trainer_callback import DefaultFlowCallback, TrainerCallback, TrainerControl, TrainerState


if TYPE_CHECKING:
    from transformers.training_args import TrainingArguments


logger = logging.getLogger(__name__)


@dataclass
class CycleTrainerState(TrainerState):
    """Extension of TrainerState to handle cycle-specific states"""

    steps_a: int = 0  # Number of updates for model A
    steps_b: int = 0  # Number of updates for model B
    loss_a: float = 0.0  # Current loss for model A
    loss_b: float = 0.0  # Current loss for model B

    @property
    def global_step(self) -> int:
        """Global step is the maximum of steps between models"""
        return max(self.steps_a, self.steps_b)

    @global_step.setter
    def global_step(self, value: int):
        """Setting global step is not allowed as it's derived from model steps"""
        raise AttributeError("global_step cannot be set directly. Update steps_a or steps_b instead.")

    def update_model_step(self, cycle_name: str):
        """Update step count for a specific model"""
        if cycle_name == "A":
            self.steps_a += 1
        elif cycle_name == "B":
            self.steps_b += 1
        else:
            raise ValueError(f"Invalid cycle name: {cycle_name}")

    def get_model_step(self, cycle_name: str) -> int:
        """Get current step for a specific model"""
        if cycle_name == "A":
            return self.steps_a
        elif cycle_name == "B":
            return self.steps_b
        else:
            raise ValueError(f"Invalid cycle name: {cycle_name}")

    def update_model_loss(self, cycle_name: str, loss: float):
        """Update current loss for a specific model"""
        if cycle_name == "A":
            self.loss_a = loss
        elif cycle_name == "B":
            self.loss_b = loss
        else:
            raise ValueError(f"Invalid cycle name: {cycle_name}")

    def get_model_loss(self, cycle_name: str) -> float:
        """Get current loss for a specific model"""
        if cycle_name == "A":
            return self.loss_a
        elif cycle_name == "B":
            return self.loss_b
        else:
            raise ValueError(f"Invalid cycle name: {cycle_name}")


class CallbackHandler(TrainerCallback):
    """Internal class that just calls the list of callbacks in order."""

    def __init__(
        self,
        callbacks: list[TrainerCallback],
        model: dict[str, Any],
        processing_class: dict[str, Any],
        optimizer: dict[str, Any],
        lr_scheduler: dict[str, Any],
    ):
        self.callbacks: list[TrainerCallback] = []
        for cb in callbacks:
            self.add_callback(cb)
        self.model = model
        self.processing_class = processing_class
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = None
        self.eval_dataloader = None

        if not any(isinstance(cb, DefaultFlowCallback) for cb in self.callbacks):
            logger.warning(
                "The Trainer will not work properly if you don't have a `DefaultFlowCallback` in its callbacks. You\n"
                + "should add one before training with `trainer.add_callback(DefaultFlowCallback). The current list of"
                + "callbacks is\n:"
                + self.callback_list
            )

    def add_callback(self, callback: TrainerCallback) -> None:
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in [c.__class__ for c in self.callbacks]:
            logger.warning(
                f"You are adding a {cb_class} to the callbacks of this Trainer, but there is already one. The current"
                + "list of callbacks is\n:"
                + self.callback_list
            )
        self.callbacks.append(cb)

    def pop_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return cb
        else:
            for cb in self.callbacks:
                if cb == callback:
                    self.callbacks.remove(cb)
                    return cb

    def remove_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return
        else:
            self.callbacks.remove(callback)

    @property
    def callback_list(self):
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    def on_init_end(self, args: "TrainingArguments", state: TrainerState, control: TrainerControl):
        return self.call_event("on_init_end", args, state, control)

    def on_train_begin(self, args: "TrainingArguments", state: TrainerState, control: TrainerControl):
        control.should_training_stop = False
        return self.call_event("on_train_begin", args, state, control)

    def on_train_end(self, args: "TrainingArguments", state: TrainerState, control: TrainerControl):
        return self.call_event("on_train_end", args, state, control)

    def on_epoch_begin(self, args: "TrainingArguments", state: TrainerState, control: TrainerControl):
        control.should_epoch_stop = False
        return self.call_event("on_epoch_begin", args, state, control)

    def on_epoch_end(self, args: "TrainingArguments", state: TrainerState, control: TrainerControl):
        return self.call_event("on_epoch_end", args, state, control)

    def on_step_begin(self, args: "TrainingArguments", state: TrainerState, control: TrainerControl):
        control.should_log = False
        control.should_evaluate = False
        control.should_save = False
        return self.call_event("on_step_begin", args, state, control)

    def on_pre_optimizer_step(self, args: "TrainingArguments", state: TrainerState, control: TrainerControl):
        return self.call_event("on_pre_optimizer_step", args, state, control)

    def on_optimizer_step(self, args: "TrainingArguments", state: TrainerState, control: TrainerControl):
        return self.call_event("on_optimizer_step", args, state, control)

    def on_substep_end(self, args: "TrainingArguments", state: TrainerState, control: TrainerControl):
        return self.call_event("on_substep_end", args, state, control)

    def on_step_end(self, args: "TrainingArguments", state: TrainerState, control: TrainerControl):
        return self.call_event("on_step_end", args, state, control)

    def on_evaluate(self, args: "TrainingArguments", state: TrainerState, control: TrainerControl, metrics):
        control.should_evaluate = False
        return self.call_event("on_evaluate", args, state, control, metrics=metrics)

    def on_predict(self, args: "TrainingArguments", state: TrainerState, control: TrainerControl, metrics):
        return self.call_event("on_predict", args, state, control, metrics=metrics)

    def on_save(self, args: "TrainingArguments", state: TrainerState, control: TrainerControl):
        control.should_save = False
        return self.call_event("on_save", args, state, control)

    def on_log(self, args: "TrainingArguments", state: TrainerState, control: TrainerControl, logs):
        control.should_log = False
        return self.call_event("on_log", args, state, control, logs=logs)

    def on_prediction_step(self, args: "TrainingArguments", state: TrainerState, control: TrainerControl):
        return self.call_event("on_prediction_step", args, state, control)

    def call_event(self, event, args, state, control, model_key="both", **kwargs):
        for callback in self.callbacks:
            if model_key == "both":
                result = getattr(callback, event)(
                    args,
                    state,
                    control,
                    model=self.model,
                    processing_class=self.processing_class,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    train_dataloader=self.train_dataloader,
                    eval_dataloader=self.eval_dataloader,
                    **kwargs,
                )
            # A Callback can skip the return of `control` if it doesn't change it.
            if result is not None:
                control = result
        return control
