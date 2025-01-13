# Multi-Adapter Cycle-Consistency Training (MACCT)

## Overview

Multi-Adapter Cycle-Consistency Training (MACCT) is an implementation of the Cycle-Consistency Training (CCT) training method using [PEFT](https://huggingface.co/docs/peft/en/index) LoRA adapters ([hu et al. 2022](https://openreview.net/forum?id=nZeVKeeFYf9)) inplace of full model weights to learn the A->B and B->A translation mappings.

### Key Benefits

- **Memory Efficient**: MACCT shares frozen base model weights, significantly reducing optimizer states
- **Larger Models**: Load a frozen base model that is `~7.5x larger` than either model in the full fine-tuning case

## Visual Explanation

<div class="video-container">
    <video controls>
        <source src="../assets/macct_animation.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

<style>
.video-container {
    position: relative;
    width: 100%;
    padding-bottom: 56.25%; /* 16:9 Aspect Ratio */
    margin: 20px 0;
}

.video-container video {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
}
</style>

## Technical Details

??? "Memory Savings Calculation"

    The memory savings are calculated by comparing static memory usage between dual-model CCT ($DMCCT$) and multi-adapter CCT ($MACCT$). We consider model weights and optimizer states, using AdamW optimizer in all cases.

    ### Variables
    - $\theta$: foundational model parameters
    - $\phi$: LoRA adapter parameters
    - $p$: bits per parameter ${\theta}_i$
    - $q$: bits per parameter ${\phi}_i$
    - $r$: ratio of base model size $|\theta|$ to LoRA adapter size $|\phi|$

    ### Formulas

    **Dual Model CCT Memory Usage:**
    $$
    \begin{aligned}
    DMCCT =& \left[ 2 \text{ models} * |\theta| \text{ params} * p \text{ bits} \right] + \left[ 2 \text{ models} * |\theta| \text{ params} * 2 \text{ states} * (4*8) \text{ bits} \right] \\
          =& 2|\theta|(p + 64)
    \end{aligned} \tag{1}
    $$

    **MACCT Memory Usage:**
    $$
    \begin{aligned}
    MACCT =& [ |\theta| \text{ params} * p \text{ bits} ] + [ 2 \text{ loras} * (|\theta| \text{ params} * r) * { q \text{ bits} + 2 \text{ states} * (4*8) \text{ bits} } ] \\
          =& |\theta|(p + 2r(q + 64))
    \end{aligned} \tag{2}
    $$

    ### Example Calculation
    With typical values:
    - $p = 16$ (base model bits)
    - $q = 32$ (LoRA trained in 32 bit)
    - $r = 0.03$ (~3% of base model size)
    - 1B parameter model for each translation in $DMCCT$

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
