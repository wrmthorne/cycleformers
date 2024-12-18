# Performance

## 1 Potential Ways to Improve Evaluation Metric Performance


lsLoRA ([Kalajdzievski, Damjan 2023](https://huggingface.co/papers/2312.03732)) scales adapters during each forward pass by `lora_alpha/math.sqrt(r)` which stabilizes performance at higher ranks $r$ ([rsLoRA docs](https://huggingface.co/papers/2312.03732)). 

DoRA ([Liu et al. 2024](https://arxiv.org/abs/2402.09353)) decomposes weight updates into magnitude and direction which they show is better correlated with full fine-tuning loss signals. This technique is particularly useful at low-ranks but can incurr a significant speed penalty. Significant performance gains can be made at the expense of higher VRAM usage by using `ephemeral_gpu_offload=True`. More info can be found at the [DoRA docs](ephemeral_gpu_offload=True).


LoRA+ ([Hayou et al. 2024](https://arxiv.org/abs/2402.12354)) is an optimisation strategy for LoRA that allows for different learning rates for adapter matrices A and B. This can increase fine-tuning speed by up to 2x and *potentially* boost performance on some tasks by 1-2% ([LoRA+ docs](https://arxiv.org/abs/2402.12354)).


## 2 Potential Ways to Improve Throughput

### 2.1 Tokenization

Wherever possible, we recommend using the same model for each direction of translation; at the very least, we recommend using models that share a tokenizer, i.e. are from the same generation of the same model family.

Sending tokens from the GPU, back to CPU to detokenize, manipulate as strings, then re-tokenizing and sending back to GPU is costly. This is particularly significant for causal models that require sequences to be split and concatenated to produce the correct input_ids for training. We can skip this overhead by manipulating tokens as tensors on the GPU if we know that the tokenizer is compatible.

Just having a compatible tokenizer is not always sufficient. If you perform any custom processing of synthetic samples before they are given to the training model such as applying a chat template, you may find that the tokenization overhead is unavoidable.

### 2.2 Specific Attention Kernels and Implementations

Flash Attention 2 ([Dao et al. 2023](https://arxiv.org/abs/2205.14135)) is a drop-in replacement for the standard attention mechanism that significantly reduces memory usage and increases throughput. It can be installed via `pip install flash-attn` (see their [github](https://github.com/Dao-AILab/flash-attention) if having installation issues) and setting `attn_implementation="flash_attention_2"` in your model config.

The Liger Kernel ([Hsu et al. 2024](https://arxiv.org/abs/2410.10989)) is a set of Triton kernels for efficient training of transformer models. They claim up to 20% throughput increase and 60% reduction in GPU memory usage. It can be installed via `pip install liger-kernel` and enabled by setting `use_liger_kernel=True` in your model config.

These methods can be combined to see a very significant throughput increase and reduction in VRAM usage.

```python
model = AutoModelForCausalLM.from_pretrained(
    # ...
    attn_implementation="flash_attention_2",
    use_liger_kernel=True
)
```

🚧 # TODO: Test whether you can compile, use optimum, etc. on base model when using MACCT.