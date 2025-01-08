from os import PathLike
from typing import Any

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, PreTrainedModel

from cycleformers.exceptions import CycleModelError


def load_model(model_path: str | PathLike[str], **model_init_kwargs: dict[str, Any]) -> PreTrainedModel:
    auto_config = AutoConfig.from_pretrained(model_path)
    if "ForCausalLM" in auto_config.model_type:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_init_kwargs)
    elif auto_config.is_encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **model_init_kwargs)
    else:
        raise CycleModelError(
            "Unsupported or unrecognised model type. Make sure the provided model is either "
            "CausalLM or Seq2SeqLM. If you are using a custom model, you may need to pass the instantiated model to "
            "CycleTrainer."
        )

    # TODO: Handle quantisation
    return model
