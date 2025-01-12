#!/bin/bash
#SBATCH --job-name=cycleformers-benchmark
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=82G
#SBATCH --time 16:00:00
#SBATCH --ntasks=1
#SBATCH --mail-user=wthorne1@sheffield.ac.uk
#SBATCH --mail-type=ALL


mkdir -p logs

ENV_NAME="cycleformers-benchmarks"
SETUP_ENV=false


module load cuDNN/8.9.2.26-CUDA-12.1.1
module load Anaconda3/2022.05

if ! conda env list | grep -q $ENV_NAME; then
    conda create -n $ENV_NAME python=3.11 -y
    SETUP_ENV=true
fi

source activate $ENV_NAME

if $SETUP_ENV; then
    python -m pip install torch
    # Or use a nightly PyTorch build instead if you want to be able to use H100 GPUs also / as well.
    # python -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu118

    pip install -e ../.

    # Test with flash-attn and liger-kernel for memory savings
    pip install ninja packaging wheel setuptools liger-kernel
    pip install flash-attn --no-build-isolation
fi

# Run the benchmark script with accelerate
accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --dynamo_backend no \
    benchmarks/experiment_benchmarks.py \
    --output_dir /path/to/output \
    --config benchmarks/configs/exp_config.yaml
