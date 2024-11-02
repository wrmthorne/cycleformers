from copy import deepcopy

import pytest
import torch


class TestAdapterGradients:
    def get_adapter_parameters(self, model, adapter_name):
        """Helper to get all parameters for a specific adapter"""
        model.set_adapter(adapter_name)
        return {
            name: param 
            for name, param in model.named_parameters() 
            if adapter_name in name
        }
    
    def make_exact_copy(self, model, adapter=None):
        """Make an exact copy of the model. Optionally set a specific adapter."""
        if adapter is not None:
            model.set_adapter(adapter)
        return deepcopy(model)

    def list_parameter_diffs(self, model_a, model_b, adapter=None):
        """Method to list weight and parameter differences between versions
        of the same model. Optionally for a specific adapter. deepcopy should
        be used to take snapshots of models."""
        if adapter is not None:
            model_a.set_adapter(adapter)
            model_b.set_adapter(adapter)
        
        a_params = model_a.named_parameters()
        b_params = model_b.named_parameters()

        diffs = {'data': [], 'grad': []}
        for (name, a_param), (_, b_param) in zip(a_params, b_params):
            if adapter is not None and adapter not in name:
                continue

            if not torch.equal(a_param, b_param):
                diffs['data'].append({
                    'name': name,
                    'a_data': a_param,
                    'b_data': b_param
                })

            if (a_param.grad is None and b_param.grad is not None or
                b_param.grad is None and a_param.grad is not None or
                not torch.equal(a_param.grad, b_param.grad)):
                diffs['grad'].append({
                    'name': name,
                    'a_grad': a_param.grad,
                    'b_grad': b_param.grad
                })
        return diffs
    
    def test_adapter_requires_grad(self, seq2seq_model, seq2seq_inputs):
        """Test that ensures all adapter parameters require gradients"""
        seq2seq_model.train()
        input_ids = seq2seq_inputs["input_ids"]
        attention_mask = seq2seq_inputs["attention_mask"]
        adapter_a_idx = seq2seq_model.adapter_to_idx["adapter_a"]
        adapter_b_idx = seq2seq_model.adapter_to_idx["adapter_b"]

        for name, param in self.get_adapter_parameters(seq2seq_model, "adapter_a").items():
            assert param.requires_grad, f"Adapter (a) parameter {name} should require gradients"
        for name, param in self.get_adapter_parameters(seq2seq_model, "adapter_b").items():
            assert param.requires_grad, f"Adapter (b) parameter {name} should require gradients"

        for (train_name, adapter_idx) in [("adapter_a", adapter_a_idx), ("adapter_b", adapter_b_idx)]:
            non_train_name = "adapter_a" if train_name == "adapter_b" else "adapter_b"
            outputs = seq2seq_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                train_adapter_idx=torch.tensor([adapter_idx]),
            )
            assert seq2seq_model.active_adapter == train_name, f"Training adapter should be {train_name}"
            outputs.loss.backward()

            for name, param in self.get_adapter_parameters(seq2seq_model, non_train_name).items():
                assert param.requires_grad, f"Non-training adapter {non_train_name} parameter {name} should require gradients"
            for name, param in self.get_adapter_parameters(seq2seq_model, train_name).items():
                assert param.requires_grad, f"Training adapter {train_name} parameter {name} should require gradients"
        

    def test_gradient_isolation(self, seq2seq_model, seq2seq_inputs):
        """
        Test that gradients only flow through the training adapter
        and other adapters remain unchanged during training.
        """        
        seq2seq_model.train()
        input_ids = seq2seq_inputs["input_ids"]
        attention_mask = seq2seq_inputs["attention_mask"]

        adapter_a_idx = seq2seq_model.adapter_to_idx["adapter_a"]
        adapter_b_idx = seq2seq_model.adapter_to_idx["adapter_b"]

        # Forward pass training adapter_a
        outputs = seq2seq_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            train_adapter_idx=torch.tensor([adapter_a_idx]),
        )
        assert seq2seq_model.active_adapter == "adapter_a", "Training adapter should be adapter_a"
        post_forward_copy_a = self.make_exact_copy(seq2seq_model, adapter="adapter_a")
        outputs.loss.backward()

        for name, param in self.get_adapter_parameters(seq2seq_model, "adapter_b").items():
            assert param.grad is None, f"Non-training adapter (b) parameter {name} should not have gradients"

        for name, param in self.get_adapter_parameters(seq2seq_model, "adapter_a").items():
            assert param.grad is not None, f"Training adapter (a) parameter {name} should have gradients"

        outputs = seq2seq_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            train_adapter_idx=torch.tensor([adapter_b_idx]),
        )
        assert seq2seq_model.active_adapter == "adapter_b", "Training adapter should be adapter_b"
        outputs.loss.backward()

        diffs = self.list_parameter_diffs(seq2seq_model, post_forward_copy_a, adapter="adapter_a")
        assert not diffs['grad'], "Non-training adapter (a) gradients should not have changed"
        assert not diffs['data'], "Non-training adapter (a) weights should not have changed"

        # Gradients should be present for training adapter
        for name, param in self.get_adapter_parameters(seq2seq_model, "adapter_b").items():
            assert param.grad is not None, f"Training adapter (b) parameter {name} should have gradients"