import pytest

from cycleformers.model_config import ModelConfig, ModelConfigA, ModelConfigB, merge_configs


class TestMergeConfigs:
    @pytest.mark.parametrize(
        "base_config, config_a, config_b, expected_a, expected_b",
        [
            # Test case 1: Basic override of model names
            (
                ModelConfig(model_name_or_path="base"),
                ModelConfigA(A_model_name_or_path="model_a"),
                ModelConfigB(B_model_name_or_path="model_b"),
                "model_a",
                "model_b",
            ),
            # Test case 2: Override with default values (should not override)
            (
                ModelConfig(lora_r=64),
                ModelConfigA(A_lora_r=16),  # default value
                ModelConfigB(B_lora_r=32),
                64,  # should keep base value since A uses default
                32,
            ),
            # Test case 3: Multiple field overrides
            (
                ModelConfig(model_name_or_path="base", lora_r=16, lora_alpha=32),
                ModelConfigA(A_model_name_or_path="model_a", A_lora_r=64, A_lora_alpha=128),
                ModelConfigB(B_model_name_or_path="model_b", B_lora_r=32, B_lora_alpha=64),
                (64, 128),  # (lora_r, lora_alpha)
                (32, 64),
            ),
        ],
        ids=["basic_override", "default_value_handling", "multiple_fields"],
    )
    def test_merge_configs_parametrized(self, base_config, config_a, config_b, expected_a, expected_b):
        result = merge_configs(base_config, config_a, config_b)

        if isinstance(expected_a, tuple):
            # Handle multiple field test case
            assert result.A.lora_r == expected_a[0]
            assert result.A.lora_alpha == expected_a[1]
            assert result.B.lora_r == expected_b[0]
            assert result.B.lora_alpha == expected_b[1]
        else:
            # Handle single field test cases
            if isinstance(expected_a, str):
                assert result.A.model_name_or_path == expected_a
                assert result.B.model_name_or_path == expected_b
            else:
                assert result.A.lora_r == expected_a
                assert result.B.lora_r == expected_b

    def test_merge_configs_preserves_base_values(self):
        """Test that unmodified base values are preserved in both A and B configs"""
        base = ModelConfig(model_name_or_path="base", lora_r=32, trust_remote_code=True)
        config_a = ModelConfigA(A_model_name_or_path="model_a")
        config_b = ModelConfigB(B_model_name_or_path="model_b")

        result = merge_configs(base, config_a, config_b)

        # Check that base values are preserved
        assert result.A.lora_r == 32
        assert result.A.trust_remote_code is True
        assert result.B.lora_r == 32
        assert result.B.trust_remote_code is True

    def test_merge_configs_list_handling(self):
        """Test handling of list values in configs"""
        base = ModelConfig(lora_target_modules=["query", "value"])
        config_a = ModelConfigA(A_lora_target_modules=["key"])
        config_b = ModelConfigB(B_lora_target_modules=["output"])

        result = merge_configs(base, config_a, config_b)

        assert result.A.lora_target_modules == ["key"]
        assert result.B.lora_target_modules == ["output"]

    def test_merge_configs_original_unmodified(self):
        """Test that original configs remain unmodified after merge"""
        base = ModelConfig(model_name_or_path="base", lora_r=32)
        config_a = ModelConfigA(A_model_name_or_path="model_a", A_lora_r=64)
        config_b = ModelConfigB(B_model_name_or_path="model_b", B_lora_r=128)

        # Store original values
        original_base_model = base.model_name_or_path
        original_base_lora_r = base.lora_r

        merge_configs(base, config_a, config_b)

        # Check original configs weren't modified
        assert base.model_name_or_path == original_base_model
        assert base.lora_r == original_base_lora_r


# Add tests here
