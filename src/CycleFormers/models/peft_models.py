from dataclasses import dataclass

import torch
from transformers import DataCollatorForSeq2Seq
from transformers.modeling_outputs import ModelOutput
from peft import PeftModelForSeq2SeqLM


@dataclass
class CycleOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    synthetic_tokens: torch.Tensor | None = None
    reconstruction_tokens: torch.Tensor | None = None


# TODO: Move later - May not be necessary
class CycleCollatorForSeq2SeqLM(DataCollatorForSeq2Seq):
    def __init__(self, *args, **kwargs):
        self.adapter_to_idx = kwargs.pop("adapter_to_idx", None)
        super().__init__(*args, **kwargs)

    def __call__(self, features):
        for feature in features:
            feature["train_adapter_idx"] = self.adapter_to_idx[feature.pop("train_adapter")]
        return super().__call__(features)
    

class PeftCycleModelForSeq2SeqLM(PeftModelForSeq2SeqLM):
    def __init__(self, model, tokenizer, peft_config, **kwargs):
        super().__init__(model, peft_config, **kwargs)
        self.tokenizer = tokenizer
        # TODO: Modify to receive dataclasses of models, etc.

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
        """
        Forward pass that handles one complete cycle. Handles an 
        """        
        if labels is not None:
            raise ValueError("Labels are not supported in this model")

        labels = input_ids.clone()
        padding_mask = (labels == self.tokenizer.pad_token_id)
        labels[padding_mask] = -100 # TODO: Make this dynamic

        # Split out the train adapter from the generation adapters
        # TODO: Need to enforce order or add ability to pass custom cycle order - would allow for multiple loops before training
        adapter_names = set(self.peft_config) # Gets names of all adapters
        if not self.idx_to_adapter[int(train_adapter_idx[0])] in adapter_names:
            raise ValueError(f"Unknown adapter index: {train_adapter_idx}. Must be one of {self.idx_to_adapter}")
        
        train_adapter_name = self.idx_to_adapter[int(train_adapter_idx[0])]
        adapter_names.remove(train_adapter_name)

        # Step 1: Generate synthetic samples with first adapter
        for adapter in adapter_names:
            self.set_adapter(adapter)
            with torch.inference_mode():
                synthetic_tokens = self.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=len(input_ids) + 100, # TODO: Make this dynamic
                    pad_token_id=self.tokenizer.pad_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

                # TODO: Add intermediate token manipulation call

        # Step 2: Generate reconstructions with second adapter
        self.set_adapter(train_adapter_name)

        synthetic_attention_mask = (synthetic_tokens != self.tokenizer.pad_token_id).long()

        # Step 3: Generate reconstructions
        outputs = super().forward(
            input_ids=synthetic_tokens,
            attention_mask=synthetic_attention_mask,
            labels=input_ids.clone(),
            **kwargs
        )

        return CycleOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            synthetic_tokens=synthetic_tokens,
            reconstruction_tokens=outputs.logits.argmax(dim=-1)
        )
