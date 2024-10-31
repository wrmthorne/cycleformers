from dataclasses import dataclass

import torch
from peft import PeftModelForSeq2SeqLM


@dataclass
class CycleOutput:
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    synthetic_tokens: torch.Tensor | None = None
    reconstruction_tokens: torch.Tensor | None = None


class PeftCycleModelForSeq2SeqLM(PeftModelForSeq2SeqLM):
    def __init__(self, model, tokenizer, peft_config, **kwargs):
        super().__init__(model, peft_config, **kwargs)
        self.tokenizer = tokenizer
        self.adapter_names = set(self.peft_config.adapter_names)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        adapter_name: str | None = None,
        **kwargs
    ):
        """
        Forward pass that handles one complete cycle. Handles an 
        """
        if labels is not None:
            raise ValueError("Labels are not supported in this model")
        if adapter_name is None:
            raise ValueError("adapter_name is required")
        if not adapter_name in self.adapter_names:
            raise ValueError(f"Unknown adapter_name: {adapter_name}. Must be one of {self.adapter_names}")

        labels = input_ids.clone()
        padding_mask = (labels == self.tokenizer.pad_token_id)
        labels[padding_mask] = -100 # TODO: Make this dynamic

        # Split out the train adapter from the generation adapters
        adapter_names = self.adapter_names.copy()
        train_adapter = adapter_names.remove(adapter_name)
        gen_adapters = adapter_names

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

        # Step 2: Generate reconstructions with second adapter
        self.set_adapter(train_adapter)

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
