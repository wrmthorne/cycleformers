import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml


yaml_base = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 2,
    "per_device_eval_batch_size": 2,
    "save_strategy": "steps",
    "max_steps": 3,
    "eval_steps": 1,
    "save_steps": 1,
    "logging_strategy": "steps",
    "logging_steps": 1,
}

lora_config = {
    "use_peft": True,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_target_modules": "all-linear",
}

causal_yaml = (
    {
        # "model_name_or_path": "Qwen/Qwen2.5-0.5B",
        "model_name_or_path": "trl-internal-testing/tiny-LlamaForCausalLM-3.1",  # TODO: Replace with QWen when tokenization skip works reliably
    }
    | yaml_base
)

seq2seq_yaml = {
    "model_name_or_path": "google/flan-t5-small",
} | yaml_base

causal_yaml_macct = {**lora_config, "lora_task_type": "CAUSAL_LM"} | causal_yaml

seq2seq_yaml_macct = {**lora_config, "lora_task_type": "SEQ_2_SEQ_LM"} | seq2seq_yaml


@pytest.fixture(scope="module")
def tmp_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.mark.parametrize("example_script", ["cycle_ner/train.py"])
@pytest.mark.parametrize(
    "config_yaml",
    [
        ("causal-base", "causal.yaml", causal_yaml),
        ("seq2seq-base", "seq2seq.yaml", seq2seq_yaml),
        ("causal-macct", "causal-macct.yaml", causal_yaml_macct),
        ("seq2seq-macct", "seq2seq-macct.yaml", seq2seq_yaml_macct),
    ],
)
def test_cycle_ner(example_script, config_yaml, tmp_dir):
    out_dirname, filename, config = config_yaml
    out_dir = Path(tmp_dir) / out_dirname

    config["output_dir"] = str(out_dir)
    yaml_file = Path(tmp_dir) / filename
    with open(yaml_file, "w") as f:
        yaml.dump(config, f)

    project_root = Path(__file__).parent.parent.parent
    command = f"python {project_root}/examples/{example_script} {yaml_file}"

    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # Check if the process completed successfully
    assert result.returncode == 0, f"Process failed with error: {result.stderr}"

    # Check for model checkpoitns
    checkpoint_dir = Path(config["output_dir"])
    assert checkpoint_dir.exists(), "Checkpoint directory was not created"
    assert any(checkpoint_dir.iterdir()), "No checkpoints were saved"
