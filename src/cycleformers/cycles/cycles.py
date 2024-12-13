import torch
import torch.nn as nn
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from cycleformers.utils import DEFAULT_SEP_SEQ


def prepare_causal_skip_cycle_inputs(
    real_input_ids: torch.Tensor,
    synth_input_ids: torch.Tensor,
    model_gen: nn.Module,
    model_train: nn.Module,
    tokenizer_gen: PreTrainedTokenizerBase,
    tokenizer_train: PreTrainedTokenizerBase,
    cycle_name: str,
    # TODO: Needs device and sep_seq
) -> BatchEncoding | dict[str, torch.Tensor]:
    """An optimised function for handling the mid-cycle token processing for causal language models, that share an
    identical tokenizer. It will be incorrect or simply crash if used in a different context.

    Because the tokenizer is identical, we can avoid the costly overhead of sending the ids back to the CPU to
    detokenize, retokenize and reship back to the GPU.

    This function accepts the prompt and the generated response along with the relevant models and tokenizers.
    We need to broadly do the following:
    1) Move the generated tokens to be the prompt and the prompt to be the response
    2) Ensure that any separating text/sequence is moved to the end of the new prompt
    3) Shift all padding tokens to the right
    4) Create attention masks and labels, handling EOS tokens properly when the pad token is the same as the eos token

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
    SEQ_SEQ_IDS = tokenizer_gen.encode(DEFAULT_SEP_SEQ)[0]
    SEP_SEQ_LEN = len(SEQ_SEQ_IDS)
    PROMPT_WIDTH = real_input_ids.shape[1]
    BATCH_SIZE, SEQ_LEN = synth_input_ids.shape
    INPUTS_WIDTH = PROMPT_WIDTH - SEP_SEQ_LEN

    device = synth_input_ids.device

    # Mask off all special tokens
    special_mask = (
        (synth_input_ids != tokenizer_gen.bos_token_id)
        & (synth_input_ids != tokenizer_gen.eos_token_id)
        & (synth_input_ids != tokenizer_gen.pad_token_id)
    ).to(device)

    # Count number of tokens in prompt and response
    prompt_lens = special_mask[:, :INPUTS_WIDTH, None].sum(dim=1).to(device)
    response_lens = special_mask[:, PROMPT_WIDTH:, None].sum(dim=1).to(device)

    # Calculate how much to shift each part
    prompt_shifts = (response_lens + SEP_SEQ_LEN).to(device)
    response_shifts = -(prompt_lens + SEP_SEQ_LEN).to(device)
    separator_shifts = -(prompt_lens - response_lens).to(device)

    # Apply shifts to indices for scatter
    indices = torch.arange(SEQ_LEN, device=device).expand(BATCH_SIZE, -1).clone()
    indices[:, :INPUTS_WIDTH] += special_mask[:, :INPUTS_WIDTH] * prompt_shifts
    indices[:, PROMPT_WIDTH:] += special_mask[:, PROMPT_WIDTH:] * response_shifts
    indices[:, INPUTS_WIDTH:PROMPT_WIDTH] += special_mask[:, INPUTS_WIDTH:PROMPT_WIDTH] * separator_shifts

    # Use BOS offset to shift padding to the right
    indices += -(synth_input_ids == tokenizer_gen.bos_token_id).nonzero().to(device)[:, -1, None]
    indices %= SEQ_LEN

    input_ids = torch.zeros_like(synth_input_ids, device=device)
    input_ids.scatter_(1, indices, synth_input_ids)

    # Right padded following scatter. EOS is always the first "pad" token
    # Only necessary if pad token is eos token
    attn_mask = input_ids != tokenizer_train.pad_token_id
    if tokenizer_train.eos_token_id == tokenizer_train.pad_token_id:
        eos_idxs = (~attn_mask & torch.roll(attn_mask, 1, dims=1)).nonzero()
        attn_mask[eos_idxs[:, 0], eos_idxs[:, 1]] = True
    else:
        eos_idxs = (synth_input_ids == tokenizer_gen.eos_token_id).nonzero()

    labels = torch.full_like(synth_input_ids, -100, device=device)
    special_mask[:, INPUTS_WIDTH:] = False
    labels.scatter_(1, indices * special_mask, synth_input_ids * special_mask)
    labels[:, 0] = -100
    labels[eos_idxs[:, 0], eos_idxs[:, 1]] = tokenizer_gen.eos_token_id

    # Clear video memory
    del (
        BATCH_SIZE,
        SEQ_LEN,
        PROMPT_WIDTH,
        INPUTS_WIDTH,
        SEP_SEQ_LEN,
        SEQ_SEQ_IDS,
        special_mask,
        prompt_lens,
        response_lens,
        prompt_shifts,
        response_shifts,
        separator_shifts,
        indices,
    )
    torch.cuda.empty_cache()

    return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}
