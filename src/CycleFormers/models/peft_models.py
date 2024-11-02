from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import DataCollatorForSeq2Seq, GenerationConfig, PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import ModelOutput
from peft import LoraConfig, PeftModelForSeq2SeqLM, get_peft_model


@dataclass
class CycleOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    synthetic_tokens: torch.Tensor | None = None
    reconstruction_tokens: torch.Tensor | None = None


@dataclass
class CycleAdapterConfig:
    adapter_name: str
    peft_config: LoraConfig | None = None
    low_cpu_mem_usage: bool = False # Strongly recommended not to set to True
    is_pretrained: bool = False
    generation_config: GenerationConfig | None = None


class CycleCollatorForSeq2SeqLM(DataCollatorForSeq2Seq):
    def __init__(self, *args, **kwargs):
        self.adapter_to_idx = kwargs.pop("adapter_to_idx", None)
        super().__init__(*args, **kwargs)

    def __call__(self, features):
        for feature in features:
            feature["train_adapter_idx"] = self.adapter_to_idx[feature.pop("train_adapter")]
        return super().__call__(features)


class PeftCycleModelForSeq2SeqLM(PeftModelForSeq2SeqLM):
    def __init__(self, model, tokenizer, adapter_configs: list[CycleAdapterConfig], **kwargs):
        super().__init__(model, peft_config=LoraConfig(), **kwargs)
        # TODO: Replace this. Hack to use class setup and not have to pass in a config
        self.delete_adapter("default")

        self.processing_class = tokenizer
        self.adapter_configs = {config.adapter_name: config for config in adapter_configs}

        for config in adapter_configs:
            if config.is_pretrained:
                self.load_adapter(config.adapter_name)
            elif config.peft_config is not None:
                self.add_adapter(config.adapter_name, config.peft_config)
            else:
                raise ValueError("PeftConfig is required for non-pretrained adapters")

    @property
    def adapter_to_idx(self):
        adapters = list(self.peft_config)
        sorted_adapters = sorted(adapters)
        return {adapter: idx for idx, adapter in enumerate(sorted_adapters)}
    
    @property
    def idx_to_adapter(self):
        adapters = list(self.peft_config)
        sorted_adapters = sorted(adapters)
        return {idx: adapter for idx, adapter in enumerate(sorted_adapters)}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        train_adapter_idx=None, # idx to pass to cuda
        labels=None,
        **kwargs
    ) -> torch.Tensor | CycleOutput:    

        labels = labels or self.prepare_labels(input_ids)

        # Split out the train adapter from the generation adapters
        adapter_names = set(self.peft_config) # Gets names of all adapters
        train_adapter_name = self.idx_to_adapter.get(int(train_adapter_idx[0]), None)
        if not train_adapter_name in adapter_names:
            raise ValueError(f"Unknown adapter index: {train_adapter_idx}. Must be one of {self.idx_to_adapter}")
        
        adapter_names.remove(train_adapter_name)
        cycle_adapters = list(adapter_names) + [train_adapter_name]
        synthetic_outputs = self.generate_synthetic_samples(
            cycle_adapters,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Step 2: Calculate reconstruction error
        self.set_adapter(train_adapter_name)
        outputs = super().forward(
            **synthetic_outputs,
            labels=input_ids.clone(),
            **kwargs
        )

        return CycleOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            synthetic_tokens=synthetic_outputs["input_ids"],
            reconstruction_tokens=outputs.logits.argmax(dim=-1)
        )
    
    def prepare_labels(self, input_ids):
        """Prepares labels from input_ids for use in the reconstruction loss."""
        labels = input_ids.clone()
        padding_mask = (labels == self.processing_class.pad_token_id)
        labels[padding_mask] = -100 # TODO: Make this dynamic
        return labels

    def prepare_tokens_for_cycle_step(self, target_adapter: str, **inputs) -> dict:
        """Prepares tokens for the next cycle step."""
        attention_mask = (inputs["input_ids"] != self.processing_class.pad_token_id).long()
        inputs["attention_mask"] = attention_mask
        return inputs

    def generate_synthetic_samples(self, cycle_adapters: list[str], **inputs):
        """Generates synthetic samples using provided adapters."""
        for current_adapter, next_adapter in zip(cycle_adapters, cycle_adapters[1:]):
            self.set_adapter(current_adapter)
            with torch.inference_mode():
                generation_config = self.adapter_configs[current_adapter].generation_config
                synthetic_outputs = self.generate(
                    **inputs,
                    generation_config=generation_config
                )
                inputs = self.prepare_tokens_for_cycle_step(next_adapter, input_ids=synthetic_outputs)
        return inputs
