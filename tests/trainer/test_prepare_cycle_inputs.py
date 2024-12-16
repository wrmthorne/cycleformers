from types import MethodType
from unittest.mock import Mock

import pytest
import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from cycleformers import DEFAULT_SEP_SEQ, CycleTrainer
from cycleformers.cycles import _default_prepare_cycle_inputs, _prepare_causal_skip_cycle_inputs


AVAILABLE_DEVICES = ["cpu"]
if torch.cuda.is_available():
    AVAILABLE_DEVICES.append("cuda")


@pytest.mark.parametrize("device", AVAILABLE_DEVICES)
class BaseTestCycleInputs:
    @pytest.fixture(name="cycle_trainer")
    def fixture_cycle_trainer(self, device):
        trainer = Mock(spec=CycleTrainer)
        accelerator = Mock(spec=Accelerator)
        accelerator.device = device

        trainer._prepare_cycle_inputs = CycleTrainer._prepare_cycle_inputs.__get__(trainer)
        trainer.sep_seq = DEFAULT_SEP_SEQ
        trainer.accelerator = accelerator
        return trainer

    @pytest.fixture(name="tokenizer")
    def fixture_tokenizer(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    # TODO: Replace these with better test examples
    @pytest.fixture(name="real_seq")
    def fixture_real_sequence(self):
        """Intentinally jagged to force padding (x10, x6)"""
        self.real_fst = 10
        self.real_snd = 6
        self.real_seq = ["A" * self.real_fst, "A" * self.real_snd]
        return self.real_seq

    @pytest.fixture(name="synth_seq")
    def fixture_synth_sequence(self):
        """Intentinally jagged to force padding (x8, x10)"""
        self.synth_fst = 8
        self.synth_snd = 10
        self.synth_seq = ["B" * self.synth_fst, "B" * self.synth_snd]
        return self.synth_seq


@pytest.mark.parametrize(
    "model_name",
    [
        "trl-internal-testing/tiny-LlamaForCausalLM-3.1",  # Llama tokenizer
        # "gpt2",  # GPT2 tokenizer
        # "facebook/opt-125m",  # OPT tokenizer (GPT based)
        # "EleutherAI/pythia-70m",  # Pythia/GPT-NeoX tokenizer (GPT based)
    ],
)
class TestPrepareCausalCycleInputs(BaseTestCycleInputs):
    @pytest.fixture(name="model")
    def fixture_model(self, model_name):
        return AutoModelForCausalLM.from_pretrained(model_name)

    @pytest.fixture(name="causal_sample")
    def fixture_prepare_causal_data(self, tokenizer, real_seq, synth_seq, sep_seq=DEFAULT_SEP_SEQ):
        """Prepare input sequences and expected outputs for causal LM testing."""
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": 512,
        }

        real_seq_with_sep = [seq + sep_seq for seq in real_seq]
        synth_seq_with_sep = [seq + sep_seq for seq in synth_seq]

        # The prompt that would be given to the generation model
        real_prompt_ids = tokenizer(real_seq_with_sep, **tokenizer_kwargs, padding_side="left").input_ids
        # PAD BOS R R R SEP

        # The output that would be generated as a continuation of the real input
        synth_seqs_eos = [synth + (tokenizer.eos_token or "") for synth in synth_seq]
        synth_response_ids = tokenizer(
            synth_seqs_eos,
            **tokenizer_kwargs,
            padding_side="right",
        ).input_ids
        # BOS S S S EOS PAD

        # Some tokenizers (like GPT2) don't add BOS automatically
        if tokenizer.bos_token_id is not None and synth_response_ids[0, 0] == tokenizer.bos_token_id:
            synth_response_ids = synth_response_ids[:, 1:]
            # S S S EOS PAD

        # The full sequence that would be returned from the causal model
        synth_output_ids = torch.cat([real_prompt_ids, synth_response_ids], dim=1)
        # PAD BOS R R R SEP S S S EOS PAD

        target_seq = [synth + real + (tokenizer.eos_token or "") for synth, real in zip(synth_seq_with_sep, real_seq)]
        targets = tokenizer(target_seq, **tokenizer_kwargs, padding_side="right")
        # BOS S S S SEP R R R EOS PAD

        labels = torch.clone(targets.input_ids)
        for i, text in enumerate(synth_seq_with_sep):
            prompt_ids = tokenizer.encode(text, **tokenizer_kwargs, padding_side=None)
            prompt_length = prompt_ids.shape[-1]
            labels[i, :prompt_length] = -100
        # Take care not to set eos token as -100
        labels[targets.attention_mask == 0] = -100

        sample_data = {
            "real_input_ids": real_prompt_ids,
            "synth_input_ids": synth_output_ids,
            "input_ids": targets.input_ids,
            "attention_mask": targets.attention_mask,
            "labels": labels,
        }
        return sample_data

    @pytest.mark.parametrize(
        "prepare_fn",
        [
            _default_prepare_cycle_inputs,
            _prepare_causal_skip_cycle_inputs,
        ],
    )
    def test_prepare_cycle_inputs(self, causal_sample, cycle_trainer, model, tokenizer, prepare_fn):
        # Copy the real method to our mock
        bound_method = MethodType(prepare_fn, cycle_trainer)
        setattr(cycle_trainer, "_prepare_cycle_inputs", bound_method)
        causal_sample = {k: v.to(cycle_trainer.accelerator.device) for k, v in causal_sample.items()}

        synth_batch = cycle_trainer._prepare_cycle_inputs(
            causal_sample["real_input_ids"],
            causal_sample["synth_input_ids"],
            model,
            model,
            tokenizer,
            tokenizer,
            "A",
        )
        # Test outputs
        assert torch.allclose(synth_batch["input_ids"], causal_sample["input_ids"])
        assert torch.allclose(synth_batch["attention_mask"], causal_sample["attention_mask"])
        assert torch.allclose(synth_batch["labels"], causal_sample["labels"])


@pytest.mark.parametrize(
    "model_name",
    [
        "google/flan-t5-small",
    ],
)
class TestPrepareSeq2SeqCycleInputs(BaseTestCycleInputs):
    @pytest.fixture(name="model")
    def fixture_model(self, model_name):
        return AutoModelForSeq2SeqLM.from_pretrained(model_name)

    @pytest.fixture(name="seq2seq_sample")
    def fixture_seq2seq_data(self, tokenizer, real_seq, synth_seq):
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": 512,
            "padding_side": "right",
        }

        # The prompt that would be given to the generation model
        real_prompt_ids = tokenizer(real_seq, **tokenizer_kwargs)
        # As if the model generated the synthetic output
        synth_response_ids = tokenizer(
            [tokenizer.pad_token + synth for synth in synth_seq],
            **tokenizer_kwargs,
        )
        targets = tokenizer(synth_seq, text_target=real_seq, **tokenizer_kwargs)

        return {
            "real_input_ids": real_prompt_ids.input_ids,
            "synth_input_ids": synth_response_ids.input_ids,
            "input_ids": targets.input_ids,
            "attention_mask": targets.attention_mask,
            "labels": targets.labels,
        }

    @pytest.mark.parametrize(
        "prepare_fn",
        [
            _default_prepare_cycle_inputs,
        ],
    )
    def test_prepare_cycle_inputs(self, seq2seq_sample, cycle_trainer, model, tokenizer, prepare_fn):
        # Copy the real method to our mock
        bound_method = MethodType(prepare_fn, cycle_trainer)
        setattr(cycle_trainer, "_prepare_cycle_inputs", bound_method)
        seq2seq_sample = {k: v.to(cycle_trainer.accelerator.device) for k, v in seq2seq_sample.items()}

        synth_batch = cycle_trainer._prepare_cycle_inputs(
            seq2seq_sample["real_input_ids"],
            seq2seq_sample["synth_input_ids"],
            model,
            model,
            tokenizer,
            tokenizer,
            "A",
        )
        assert torch.allclose(synth_batch["input_ids"], seq2seq_sample["input_ids"])
        assert torch.allclose(synth_batch["attention_mask"], seq2seq_sample["attention_mask"])
        assert torch.allclose(synth_batch["labels"], seq2seq_sample["labels"])
