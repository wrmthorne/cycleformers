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
    """A PEFT model wrapper that handles the logic for cycle training.

    This model extends PeftModelForSeq2SeqLM to support cycle training, where multiple adapters are used in sequence
    to generate synthetic samples and reconstructions. One adapter is trained while others are used for inference.
    The model can support an arbitrary number of adapters per cycle. The model is designed to be used with the
    CycleTrainer.

    Args:
        model (:obj:`PreTrainedModel`): The base model to apply PEFT adapters to
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer associated with the model
        peft_config (:obj:`PeftConfig`): The PEFT configuration for the adapters
        **kwargs: Additional keyword arguments passed to PeftModelForSeq2SeqLM

    WARNING: This API will change in the future.

    Example::

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from peft import LoraConfig
        from CycleFormers.models import PeftCycleModelForSeq2SeqLM

        # Initialize base model and tokenizer
        base_model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        tokenizer = AutoTokenizer.from_pretrained("t5-base")

        # Create PEFT config
        peft_config = LoraConfig(
            task_type="SEQ_2_SEQ_LM",
            r=8,
            lora_alpha=32,
            target_modules=["q", "v"]
        )

        # Create cycle model with two adapters
        model = PeftCycleModelForSeq2SeqLM(
            model=base_model,
            tokenizer=tokenizer, 
            peft_config=peft_config,
            adapter_name="adapter_a"
        )
        model.add_adapter("adapter_b", peft_config)

        # Prepare inputs
        inputs = tokenizer("Translate to French: Hello world!", return_tensors="pt")
        
        # Forward pass with adapter_a as training adapter
        outputs = model(
            **inputs,
            train_adapter_idx=torch.tensor([0])  # adapter_a's index
        )
    """

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
        """Forward pass for cycle training.

        This method implements the cycle training logic:
        1. Uses non-training adapters to generate synthetic samples from the input
        2. Uses the training adapter to reconstruct the original input from the synthetic samples
        3. Computes the loss between reconstructions and original input

        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Input token ids.
            attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid attention on padding token indices.
            train_adapter_idx (:obj:`torch.LongTensor` of shape :obj:`(1,)`):
                Index of the adapter to train. Other adapters will be used for generation.
            labels (:obj:`torch.LongTensor`, `optional`):
                Not supported - will raise ValueError if provided.
            **kwargs: Additional arguments passed to the base model's forward method.

        Returns:
            :class:`~CycleOutput`: A dataclass containing:
                - loss (:obj:`torch.FloatTensor`, `optional`): The reconstruction loss
                - logits (:obj:`torch.FloatTensor`): The model's output logits
                - synthetic_tokens (:obj:`torch.LongTensor`): Generated synthetic tokens
                - reconstruction_tokens (:obj:`torch.LongTensor`): Generated reconstruction tokens

        Raises:
            ValueError: If labels are provided or if train_adapter_idx is invalid
        """     
        if labels is not None:
            raise ValueError("Labels are not supported in this model. They are calculated internally from the input_ids.")

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