## 🚧 Usage documentation still under construction


## Multi-Adapter Cycle-Consistency Training (MACCT)

Multi-Adapter Cycle-Consistency Training (MACCT) is an implementation of the Cycle-Consistency Training (CCT) training method using [PEFT](https://huggingface.co/docs/peft/en/index) LoRA adapters ([hu et al. 2022](https://openreview.net/forum?id=nZeVKeeFYf9)) inplace of full model weights to learn the A->B and B->A translation mappings. MACCT shares the base model weights which are frozen; therefore, we greatly reduce the number of optimizer states, making a significant reduction in memory footprint. So much so, that we can load a frozen base model that is ~7.5x larger than either model in the full fine-tuning case.

[A HELPFUL DIAGRAM WILL GO HERE]


<details>
<summary><b>Click to see these figures were calculated</b></summary>

</br>

Assuming a restricted case where we just look at static memory, i.e. model weights and optimizer states, the memory savings are significant. We will assume in the dual-model ($DMCCT$) case that both models are the same, likewise in multi-adapter CCT ($MACCT$) we assume both LoRA adapters are initialised the same. In all cases we use the AdamW optimizer as it is the most popular:

With $\theta$ as the foundational model parameters, $\phi$ as the LoRA adapter parameters, $p$ as the number of bits per parameter ${\theta}_i$, $q$ as the number of bits per parameter ${\phi}_i$, and $r$ as the ratio of base model size $|\theta|$ to LoRA adapter size $|\phi|$, we can derive the memory savings as follows: 

$$
\begin{aligned}
DMCCT =& \left[ 2 \text{ models} * |\theta| \text{ params} * p \text{ bits} \right] + \left[ 2 \text{ models} * |\theta| \text{ params} * 2 \text{ states} * (4*8) \text{ bits} \right] \\
      =& 2|\theta|(p + 64)
\end{aligned} \tag{1}
$$

$$
\begin{aligned}
MACCT =& [ |\theta| \text{ params} * p \text{ bits} ] + [ 2 \text{ loras} * (|\theta| \text{ params} * r) * { q \text{ bits} + 2 \text{ states} * (4*8) \text{ bits} } ] \\
      =& |\theta|(p + 2r(q + 64))
\end{aligned} \tag{2}
$$

Assuming $p = 16$, $q = 32$ (LoRA are trained in 32 bit by default) and $r = 0.03$ (~3% of base model size), and a 1B parameter model for each translation in $DMCCT$:

$$
\begin{aligned}
    DMCCT =& 2 * 1e9 * (16 + 64) \\
          =& 1.6e11 \text{ bits} \\
    \\
    DMCCT =& MACCT \\
    1.6e11 \text{ bits} =& N * (16 + 2 * 0.03 * (32 + 64)) \\
    N =& \frac{1.6e11}{21.76} \\
      \approx& 7.35B \text{ params} \\
\end{aligned}
$$
</details>


### Getting the Best Performance

#### Tokenization

Use models with a compatible tokenizer e.g. the same model or a smaller and larger version of the same model. Sending tokens from the GPU, back to CPU to detokenize, manipulate as strings, then re-tokenizing and sending back to GPU is costly. If the same tokenizer is applicable for both models, the tokens can be manipulated while still as tensors on the GPU, reducing overhead.

However, this cost may be unavoidable such as in situations where chat templates are used.
