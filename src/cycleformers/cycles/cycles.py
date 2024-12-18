import gc
from copy import copy

import torch
import torch.nn as nn
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase


def _default_prepare_cycle_inputs(
    self,
    real_input_ids: torch.Tensor,
    synth_input_ids: torch.Tensor,
    model_gen: nn.Module,
    model_train: nn.Module,
    tokenizer_gen: PreTrainedTokenizerBase,
    tokenizer_train: PreTrainedTokenizerBase,
    cycle_name: str,
) -> BatchEncoding | dict[str, torch.Tensor]:
    """Default implementation for preparing cycle inputs during cycle-consistency training.

    This function handles the core logic of preparing inputs for the cycle training loop. It processes both
    encoder-decoder (seq2seq) and decoder-only (causal) model architectures, though some combinations are not
    yet supported.

    The function performs the following key steps:
    1. For causal models, trims the synthetic input ids to remove prompt tokens
    2. Decodes the synthetic tokens back to text
    3. For causal models, combines synthetic text with original input using a separator
    4. Tokenizes the processed text for the training model
    5. Handles label creation differently for seq2seq vs causal models:
        - For seq2seq: Uses original input ids directly as labels
        - For causal: Creates labels with -100 for prompt tokens and padding

    Args:
        real_input_ids (torch.Tensor): Original input token IDs from the training data
        synth_input_ids (torch.Tensor): Generated token IDs from the generative model
        model_gen (nn.Module): The model used for generation (source of synth_input_ids)
        model_train (nn.Module): The model being trained in this cycle
        tokenizer_gen (PreTrainedTokenizerBase): Tokenizer for the generative model
        tokenizer_train (PreTrainedTokenizerBase): Tokenizer for the training model
        cycle_name (str): Identifier for the current cycle ('A' or 'B')

    Returns:
        BatchEncoding | dict[str, torch.Tensor]: Processed inputs ready for model training, including:
            - input_ids: Token IDs for model input
            - attention_mask: Attention mask for input tokens
            - labels: Target labels for training (-100 for masked positions)

    Note:
        Currently does not support cross-architecture cycles (causal-to-seq2seq or seq2seq-to-causal).
        Future implementations may add support for these combinations.
    """
    device = self.accelerator.device

    if not model_gen.config.is_encoder_decoder:
        synth_input_ids = synth_input_ids[:, real_input_ids.shape[-1] :]

    synth_batch_text = tokenizer_gen.batch_decode(synth_input_ids, skip_special_tokens=True)

    if not model_train.config.is_encoder_decoder:
        input_texts = tokenizer_train.batch_decode(real_input_ids, skip_special_tokens=True)
        # TODO: Investigate tokenizer_train.eos_token as separator to appear more like packed training instances
        # TODO: Potentially add a configurable templating function here
        synth_batch_responses = copy(synth_batch_text)
        synth_batch_text = [
            synth_text + self.sep_seq + input_text.strip(self.sep_seq) + tokenizer_train.eos_token
            for synth_text, input_text in zip(synth_batch_text, input_texts)
        ]

    # Temporarily call padding_side in tokenizer to ensure position ids are correct for loss calculation
    synth_batch = tokenizer_train(
        synth_batch_text,
        return_tensors="pt",
        padding=True,
        padding_side="right" if not model_train.config.is_encoder_decoder else tokenizer_train.padding_side,
    ).to(device)

    # Seq2seq just needs the real input ids as labels
    synth_batch["labels"] = real_input_ids

    # FIXME: Temporary solution to fix incorrect labelling - come back and
    # implement a more efficient solution
    if not model_train.config.is_encoder_decoder:
        prompt_mask = torch.zeros_like(synth_batch["input_ids"])
        for i, text in enumerate(synth_batch_responses):
            prompt_len = (
                tokenizer_train(
                    text + self.sep_seq,
                    return_tensors="pt",
                    padding=False,
                )
                .input_ids.to(device)
                .shape[-1]
            )
            prompt_mask[i, :prompt_len] = 1

        synth_batch["labels"] = synth_batch["input_ids"].clone().to(device)  # Careful not to set eos token as -100
        synth_batch["labels"][(synth_batch["attention_mask"] == 0) | (prompt_mask == 1)] = -100

    return synth_batch


def _prepare_seq2seq_cycle_inputs(
    self,
    real_input_ids: torch.Tensor,
    synth_input_ids: torch.Tensor,
    model_gen: nn.Module,
    model_train: nn.Module,
    tokenizer_gen: PreTrainedTokenizerBase,
    tokenizer_train: PreTrainedTokenizerBase,
    cycle_name: str,
) -> BatchEncoding | dict[str, torch.Tensor]:
    """Prepare input sequences for seq2seq language models."""
    raise NotImplementedError("Too much effort for very little gain at this point")


def _prepare_causal_skip_cycle_inputs(
    self,
    real_input_ids: torch.Tensor,
    synth_input_ids: torch.Tensor,
    model_gen: nn.Module,
    model_train: nn.Module,
    tokenizer_gen: PreTrainedTokenizerBase,
    tokenizer_train: PreTrainedTokenizerBase,
    cycle_name: str,
) -> BatchEncoding | dict[str, torch.Tensor]:
    """An optimised function for handling the mid-cycle token processing for causal language models, that share an
    identical tokenizer. It will be incorrect or simply crash if used in a different context.

    Because the tokenizer is identical, we can avoid the costly overhead of sending the ids back to the CPU to
    detokenize, retokenize and reship back to the GPU.

    This function accepts the prompt and the generated response along with the relevant models and tokenizers.
    We need to broadly do the following:
    1) Move the generated tokens to be the prompt and the prompt to be the response
    2) Ensure that any separating text/sequence is moved to the end of the new prompt and before the response
    3) Shift all padding tokens to the right
    4) Create attention masks and labels, handling EOS tokens properly when the pad token is the same as the eos token

    CURRENTLY VALIDATED ON LLaMa-3.x and Qwen-2.5

    Args:
        real_input_ids (torch.Tensor): Batch of prompt token IDs [batch_size, prompt_width]
        synth_input_ids (torch.Tensor): Batch of generated sequence token IDs [batch_size, seq_len]
        model_gen: The generative model
        model_train: The model being trained
        tokenizer_gen (PreTrainedTokenizerBase): Tokenizer for the generative model
        tokenizer_train (PreTrainedTokenizerBase): Tokenizer for the training model
        cycle_name (str): Name of the cycle being trained

    Returns:
        dict: Dictionary containing input_ids, attention_mask and labels tensors

    Example:
        >>> import torch
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> tokenizer.pad_token = tokenizer.eos_token
        >>> real = torch.tensor([[1, 2, 3, 50256]])  # [batch=1, prompt_width=4]
        >>> synth = torch.tensor([[1, 4, 5, 6, 50256, 0, 0]])  # [batch=1, seq_len=7]
        >>> output = _prepare_cycle_inputs_causal_skip(
        ...     real, synth, None, None, tokenizer, tokenizer, "A"
        ... )
        >>> list(output.keys())
        ['input_ids', 'attention_mask', 'labels']
        >>> output['input_ids'].shape
        torch.Size([1, 7])
        >>> output['attention_mask'].shape
        torch.Size([1, 7])
        >>> output['labels'].shape
        torch.Size([1, 7])
    """
    SEP_SEQ_IDS = tokenizer_gen.encode(self.sep_seq, padding=False)
    if SEP_SEQ_IDS[0] == tokenizer_gen.bos_token_id:
        SEP_SEQ_IDS = SEP_SEQ_IDS[1:]
    SEP_SEQ_LEN = len(SEP_SEQ_IDS)
    PROMPT_WIDTH = real_input_ids.shape[1]
    BATCH_SIZE, SEQ_LEN = synth_input_ids.shape
    INPUTS_WIDTH = PROMPT_WIDTH - SEP_SEQ_LEN

    device = self.accelerator.device

    # Mask off all special tokens
    special_mask = torch.ones_like(synth_input_ids).to(device)
    if tokenizer_gen.bos_token_id is not None:
        special_mask &= ~torch.eq(synth_input_ids, tokenizer_gen.bos_token_id)
    if tokenizer_gen.eos_token_id is not None:
        special_mask &= ~torch.eq(synth_input_ids, tokenizer_gen.eos_token_id)
    if tokenizer_gen.pad_token_id is not None:
        special_mask &= ~torch.eq(synth_input_ids, tokenizer_gen.pad_token_id)

    # Count number of tokens in prompt and response
    prompt_lens = torch.sum(special_mask[:, :INPUTS_WIDTH, None], dim=1).to(device)
    response_lens = torch.sum(special_mask[:, PROMPT_WIDTH:, None], dim=1).to(device)

    # Calculate how much to shift each part
    prompt_shifts = (response_lens + SEP_SEQ_LEN).to(device)
    response_shifts = -(prompt_lens + SEP_SEQ_LEN).to(device)
    separator_shifts = -(prompt_lens - response_lens).to(device)

    # Apply shifts to indices for scatter
    indices = torch.arange(SEQ_LEN, device=device).expand(BATCH_SIZE, -1).clone()
    indices[:, :INPUTS_WIDTH] += special_mask[:, :INPUTS_WIDTH] * prompt_shifts
    indices[:, PROMPT_WIDTH:] += special_mask[:, PROMPT_WIDTH:] * response_shifts
    indices[:, INPUTS_WIDTH:PROMPT_WIDTH] += special_mask[:, INPUTS_WIDTH:PROMPT_WIDTH] * separator_shifts

    # Use first non-special token as offset. +1 to offset if BOS token present
    first_content_indices = -special_mask.to(torch.int64).argmax(dim=1)
    indices += first_content_indices[:, None] + bool(tokenizer_gen.bos_token_id)

    # Mod by length to wrap value around - cannot have negatives for scatter
    indices %= SEQ_LEN
    input_ids = torch.zeros_like(synth_input_ids, device=device)
    input_ids.scatter_(1, indices, synth_input_ids)

    # Right padded following scatter. EOS is always the first "PAD" token
    # Only necessary if PAD token is EOS token
    attn_mask = input_ids != tokenizer_train.pad_token_id
    if tokenizer_train.eos_token_id == tokenizer_train.pad_token_id:
        eos_idxs = (~attn_mask & torch.roll(attn_mask, 1, dims=1)).nonzero()
        attn_mask[eos_idxs[:, 0], eos_idxs[:, 1]] = True
    else:
        eos_idxs = torch.eq(synth_input_ids, tokenizer_gen.eos_token_id).nonzero()

    labels = torch.full_like(synth_input_ids, -100, device=device)
    special_mask[:, INPUTS_WIDTH:] = False
    labels.scatter_(1, indices * special_mask, synth_input_ids * special_mask)
    labels[:, 0] = -100  # TODO: Need to understand why col 0 won't stay as -100
    labels[eos_idxs[:, 0], eos_idxs[:, 1]] = tokenizer_gen.eos_token_id

    # Clear video memory
    del (
        BATCH_SIZE,
        SEQ_LEN,
        PROMPT_WIDTH,
        INPUTS_WIDTH,
        SEP_SEQ_LEN,
        SEP_SEQ_IDS,
        special_mask,
        prompt_lens,
        response_lens,
        prompt_shifts,
        response_shifts,
        separator_shifts,
        first_content_indices,
        indices,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {"input_ids": input_ids, "attention_mask": attn_mask.to(torch.int64), "labels": labels}
