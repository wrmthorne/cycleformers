from unittest.mock import Mock

import pytest

from cycleformers import CycleTrainer


class TestCyclePrepareInputs:
    @pytest.fixture(name="real_causal_input_ids")
    def fixture_real_causal_input_ids(self, causal_tokenizer):
        """Intentinally jagged to force padding (x10, x6)"""
        self.real_fst = 10
        self.real_snd = 6
        input_ids = causal_tokenizer(["A" * self.real_fst, "A" * self.real_snd], return_tensors="pt", padding=True)[
            "input_ids"
        ]
        return input_ids

    @pytest.fixture(name="synth_causal_input_ids")
    def fixture_synth_causal_input_ids(self, causal_tokenizer):
        """Intentinally jagged to force padding (x8, x10)"""
        self.synth_fst = 8
        self.synth_snd = 10
        input_ids = causal_tokenizer(["B" * self.synth_fst, "B" * self.synth_snd], return_tensors="pt", padding=True)[
            "input_ids"
        ]
        return input_ids

    @pytest.fixture(name="cycle_trainer")
    def fixture_cycle_trainer(self):
        trainer = Mock(spec=CycleTrainer)
        # Copy the real method to our mock
        trainer._cycle_prepare_inputs = CycleTrainer._cycle_prepare_inputs.__get__(trainer)
        return trainer

    @pytest.mark.skip(reason="Not implemented")
    def test_causal_causal_sequence_order(
        self, real_causal_input_ids, synth_causal_input_ids, cycle_trainer, causal_base_model, causal_tokenizer
    ):
        synth_batch = cycle_trainer._cycle_prepare_inputs(
            real_causal_input_ids,
            synth_causal_input_ids,
            causal_base_model,
            causal_base_model,
            causal_tokenizer,
            causal_tokenizer,
            "a",
        )
        assert synth_batch["input_ids"][0].tolist()[: self.synth_fst] == synth_causal_input_ids[0].tolist()
        assert synth_batch["input_ids"][1].tolist()[: self.synth_snd] == synth_causal_input_ids[1].tolist()
        assert (
            synth_batch["input_ids"][0].tolist()[self.synth_fst : self.synth_fst + self.real_fst]
            == real_causal_input_ids[0].tolist()
        )
