from copy import deepcopy

import pytest
import torch


# TODO: Break up into isolated test cases with test factories
@pytest.mark.slow
class TestParameterUpdates:
    def get_adapter_parameters(self, model, adapter_name):
        """Helper to get all parameters for a specific adapter"""
        model.set_adapter(adapter_name)
        return {
            name: param 
            for name, param in model.named_parameters() 
            if adapter_name in name
        }
    
    def make_exact_copy(self, model, adapter=None):
        """Make an exact copy of the model. Optionally set a specific adapter to assign as active."""
        if adapter is not None:
            model.set_adapter(adapter)

        model_copy = deepcopy(model)
        for param_orig, param_new in zip(model.parameters(), model_copy.parameters()):
            if param_orig.grad is not None:
                param_new.grad = param_orig.grad.clone()
            param_new.requires_grad = param_orig.requires_grad
        return model_copy
    
    @pytest.mark.meta
    def test_make_exact_copy(self, seq2seq_model, seq2seq_inputs):
        """Test that the copy is an exact copy of the model"""
        seq2seq_model.train()
        adapter_a_idx = seq2seq_model.adapter_to_idx["adapter_a"]

        outputs = seq2seq_model(
            **seq2seq_inputs,
            train_adapter_idx=torch.tensor([adapter_a_idx]),
        )
        outputs.loss.backward()
        copy_model = self.make_exact_copy(seq2seq_model, adapter="adapter_a")

        diffs = self.list_parameter_diffs(seq2seq_model, copy_model, adapter="adapter_a")
        assert not diffs['grad'], "Gradients should not have changed"
        assert not diffs['data'], "Weights should not have changed"

        outputs = seq2seq_model(
            **seq2seq_inputs,
            train_adapter_idx=torch.tensor([adapter_a_idx]),
        )

        diffs = self.list_parameter_diffs(seq2seq_model, copy_model, adapter="adapter_a")
        assert diffs['grad'], "Gradients should have have changed"
        assert not diffs['data'], "Weights should not have changed"

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
            if a_param.grad is None and b_param.grad is None:
                continue
            elif ((a_param.grad is None and b_param.grad is not None) or
                  (b_param.grad is None and a_param.grad is not None) or
                  (not torch.equal(a_param.grad, b_param.grad))):
                diffs['grad'].append({
                    'name': name,
                    'a_grad': a_param.grad,
                    'b_grad': b_param.grad
                })
        return diffs
    
    def get_param_group_idxs(self, optimizer_instance, named_parameters):
        unique_param_groups = set()
        # Use memory address of parameter to find which groups it belongs to
        for param in named_parameters.values():
            param_ptr = param.data_ptr()
            found_group = False
            for group_idx, group in enumerate(optimizer_instance.param_groups):
                for p in group['params']:
                    if param_ptr == p.data_ptr():
                        unique_param_groups.add(group_idx)
                        found_group = True
                        break
                if found_group:
                    break
            if not found_group:
                unique_param_groups.add(None)
        return unique_param_groups
    
    def test_adapter_requires_grad(self, seq2seq_model, seq2seq_inputs):
        """Test that ensures all adapter parameters require gradients"""
        seq2seq_model.train()
        input_ids = seq2seq_inputs["input_ids"]
        attention_mask = seq2seq_inputs["attention_mask"]
        adapter_a_idx = seq2seq_model.adapter_to_idx["adapter_a"]
        adapter_b_idx = seq2seq_model.adapter_to_idx["adapter_b"]

        # Fresh initialization
        for name, param in self.get_adapter_parameters(seq2seq_model, "adapter_a").items():
            assert param.requires_grad, f"Adapter (a) parameter {name} should require gradients"
            assert param.grad is None, f"Adapter (a) parameter {name} should not have gradients"
        for name, param in self.get_adapter_parameters(seq2seq_model, "adapter_b").items():
            assert param.requires_grad, f"Adapter (b) parameter {name} should require gradients"
            assert param.grad is None, f"Adapter (b) parameter {name} should not have gradients"

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

    def test_base_model_frozen(self, seq2seq_model):
        """Test that the base model is frozen"""
        seq2seq_model.train()
        for name, param in seq2seq_model.named_parameters():
            if not any(adapter in name for adapter in ["adapter_a", "adapter_b"]):
                assert not param.requires_grad, f"Base model parameter {name} should not require gradients"

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
        outputs.loss.backward()
        post_forward_copy_a = self.make_exact_copy(seq2seq_model, adapter="adapter_a")

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

        diffs = self.list_parameter_diffs(post_forward_copy_a, seq2seq_model, adapter="adapter_a")
        assert not diffs['grad'], "Non-training adapter (a) gradients should not have changed"
        assert not diffs['data'], "Non-training adapter (a) weights should not have changed"

        # Gradients should be present for training adapter
        for name, param in self.get_adapter_parameters(seq2seq_model, "adapter_b").items():
            assert param.grad is not None, f"Training adapter (b) parameter {name} should have gradients"

    # TODO: Future feature test for separate parameter groups for each adapter
    def test_optimizer_init_single_param_group(self, seq2seq_model, optimizer):
        """Test that the optimizer is initialized correctly"""
        model = seq2seq_model
        optimizer = optimizer(model.parameters())

        param_group_a_idxs = self.get_param_group_idxs(optimizer, self.get_adapter_parameters(model, "adapter_a"))
        param_group_b_idxs = self.get_param_group_idxs(optimizer, self.get_adapter_parameters(model, "adapter_b"))
        assert len(param_group_a_idxs) == 1, "Adapter (a) should only be in one parameter group"
        assert len(param_group_b_idxs) == 1, "Adapter (b) should only be in one parameter group"
        assert param_group_a_idxs == param_group_b_idxs, "Adapter (a) and (b) should be in the same parameter group"

    # TODO: Needs trainer test for gradient accumulation
    def test_parameter_updates(self, seq2seq_model, seq2seq_inputs, optimizer):
        """Test that parameters are updated correctly during training. Gradients are accumulated every 2 steps
        to have non-None gradients for both adapters to make sure step and zero_grad are working."""
        model = seq2seq_model
        optimizer = optimizer(model.parameters())
        model.train()

        adapter_a_idx = model.adapter_to_idx["adapter_a"]
        adapter_b_idx = model.adapter_to_idx["adapter_b"]

        for idx, (train_name, adapter_idx) in enumerate([("adapter_a", adapter_a_idx), ("adapter_b", adapter_b_idx)]):
            non_train_name = "adapter_a" if train_name == "adapter_b" else "adapter_b"
        
            outputs = model(
                **seq2seq_inputs,
                train_adapter_idx=torch.tensor([adapter_idx]),
            )
            outputs.loss.backward()

            train_copy = self.make_exact_copy(seq2seq_model, adapter=train_name)
            non_train_copy = self.make_exact_copy(seq2seq_model, adapter=non_train_name)

            if idx % 2 == 0:
                optimizer.step()

            diffs = self.list_parameter_diffs(non_train_copy, seq2seq_model, adapter=non_train_name)
            if idx % 2 == 0:
                assert not diffs['grad'], f"Non-training adapter {non_train_name} gradients should not have changed"
                assert not diffs['data'], f"Non-training adapter {non_train_name} weights should not have changed"
            else:
                assert diffs['grad'], f"Non-training adapter {non_train_name} gradients should have changed"
                assert diffs['data'], f"Non-training adapter {non_train_name} weights should have changed"

            diffs = self.list_parameter_diffs(train_copy, seq2seq_model, adapter=train_name)
            if idx % 2 == 0:
                assert not diffs['grad'], f"Training adapter {train_name} gradients should not have changed"
                assert diffs['data'], f"Training adapter {train_name} weights should have changed"
            else:
                assert diffs['grad'], f"Training adapter {train_name} gradients should have changed"
                assert diffs['data'], f"Training adapter {train_name} weights should have changed"

            if idx % 2 == 0:
                optimizer.zero_grad()
            
            for name, param in self.get_adapter_parameters(model, train_name).items():
                if idx % 2 == 0:
                    assert param.grad is None, f"Training adapter {train_name} parameter {name} should have None gradient"
                else:
                    assert param.grad is not None, f"Training adapter {train_name} parameter {name} should have non-None gradient"
            
            for name, param in self.get_adapter_parameters(model, non_train_name).items():
                if idx % 2 == 0:
                    assert param.grad is None, f"Non-training adapter {non_train_name} parameter {name} should have None gradient"
                else:
                    assert param.grad is not None, f"Non-training adapter {non_train_name} parameter {name} should have non-None gradient"

