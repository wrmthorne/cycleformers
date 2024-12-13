from unittest.mock import Mock

import pytest
import torch

from cycleformers import DEFAULT_SEP_SEQ, CycleTrainer


class TestCyclePrepareInputs:
    @pytest.fixture(name="real_seq")
    def fixture_real_sequence(self):
        """Intentinally jagged to force padding (x10, x6)"""
        self.real_fst = 10
        self.real_snd = 6
        self.real_seq = ["A" * self.real_fst + DEFAULT_SEP_SEQ, "A" * self.real_snd + DEFAULT_SEP_SEQ]
        return self.real_seq

    @pytest.fixture(name="synth_seq")
    def fixture_synth_sequence(self):
        """Intentinally jagged to force padding (x8, x10)"""
        self.synth_fst = 8
        self.synth_snd = 10
        self.synth_seq = ["B" * self.synth_fst, "B" * self.synth_snd]
        return self.synth_seq

    @pytest.fixture(name="real_seq2seq_input_ids")
    def fixture_real_seq2seq_input_ids(self, seq2seq_tokenizer, real_seq):
        return seq2seq_tokenizer(real_seq, return_tensors="pt", padding=True).input_ids

    @pytest.fixture(name="real_causal_input_ids")
    def fixture_real_causal_input_ids(self, causal_tokenizer, real_seq):
        return causal_tokenizer(real_seq, return_tensors="pt", padding=True).input_ids

    # FIXME: flant5 doesn't have a BOS token but other seq2seq models that will be tested do
    @pytest.fixture(name="synth_seq2seq_input_ids")
    def fixture_synth_seq2seq_input_ids(self, seq2seq_tokenizer, synth_seq):
        return seq2seq_tokenizer(synth_seq, return_tensors="pt", padding=True).input_ids

    @pytest.fixture(name="synth_causal_input_ids")
    def fixture_synth_causal_input_ids(self, causal_tokenizer, synth_seq):
        return causal_tokenizer(
            synth_seq,
            return_tensors="pt",
            padding=True,
            padding_side="right",
        ).input_ids[:, 1:]  # Remove Bos token

    @pytest.fixture(name="causal_targets")
    def fixture_causal_targets(self, causal_tokenizer, real_seq, synth_seq):
        """
        targets: {
            'input_ids': tensor([[128000,  74542,  74542,    271,  59005,   6157, 128009, 128009],
                                [128000,  74542,  74542,  10306,    271,  26783,   6157, 128009]]),
            'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 0],
                                     [1, 1, 1, 1, 1, 1, 1, 1]]),
            'labels': tensor([[-100, -100, -100, -100, 59005, 6157, 128009, -100],
                               [-100, -100, -100, -100, -100, 26783, 6157, 128009]])
        }
        """
        prompt_seqs = [synth_seq[0] + DEFAULT_SEP_SEQ, synth_seq[1] + DEFAULT_SEP_SEQ]
        sequence = [
            prompt_seqs[0] + real_seq[0].replace(DEFAULT_SEP_SEQ, "") + causal_tokenizer.eos_token,
            prompt_seqs[1] + real_seq[1].replace(DEFAULT_SEP_SEQ, "") + causal_tokenizer.eos_token,
        ]

        targets = causal_tokenizer(sequence, return_tensors="pt", padding=True, padding_side="right")
        labels = torch.clone(targets.input_ids)
        for i in range(len(prompt_seqs)):
            prompt_ids = causal_tokenizer.encode(prompt_seqs[i], return_tensors="pt", padding=False)
            prompt_length = prompt_ids.shape[-1]
            labels[i, :prompt_length] = -100

        # Need to be careful not to set eos token as -100
        labels[targets.attention_mask == 0] = -100
        targets["labels"] = labels
        return targets

    @pytest.fixture(name="seq2seq_targets")
    def fixture_seq2seq_targets(self, seq2seq_tokenizer, real_seq2seq_input_ids, synth_seq):
        targets = seq2seq_tokenizer(synth_seq, return_tensors="pt", padding=True)
        targets["labels"] = real_seq2seq_input_ids
        return targets

    @pytest.fixture(name="cycle_trainer")
    def fixture_cycle_trainer(self):
        trainer = Mock(spec=CycleTrainer)
        # Copy the real method to our mock
        trainer._cycle_prepare_inputs = CycleTrainer._prepare_cycle_inputs.__get__(trainer)
        trainer.sep_seq = DEFAULT_SEP_SEQ
        return trainer

    def test_causal_correct_input_ids(
        self,
        real_causal_input_ids,
        synth_causal_input_ids,
        causal_targets,
        cycle_trainer,
        causal_model,
        causal_tokenizer,
    ):
        synth_batch = cycle_trainer._cycle_prepare_inputs(
            real_causal_input_ids,
            torch.cat([real_causal_input_ids, synth_causal_input_ids], dim=1),
            causal_model,
            causal_model,
            causal_tokenizer,
            causal_tokenizer,
            "a",
        )
        assert torch.allclose(synth_batch["input_ids"], causal_targets["input_ids"])
        assert torch.allclose(synth_batch["labels"], causal_targets["labels"])

    def test_seq2seq_correct_input_ids(
        self,
        real_seq2seq_input_ids,
        synth_seq2seq_input_ids,
        seq2seq_targets,
        cycle_trainer,
        seq2seq_model,
        seq2seq_tokenizer,
    ):
        synth_batch = cycle_trainer._cycle_prepare_inputs(
            real_seq2seq_input_ids,
            synth_seq2seq_input_ids,
            seq2seq_model,
            seq2seq_model,
            seq2seq_tokenizer,
            seq2seq_tokenizer,
            "a",
        )
        assert torch.allclose(synth_batch["input_ids"], seq2seq_targets["input_ids"])
        assert torch.allclose(synth_batch["labels"], seq2seq_targets["labels"])
