import pytest
import yaml
from transformers import AutoConfig, PretrainedConfig

from tests.testing_utils.model_registry import (
    CapabilityExpression,
    ModelCapability,
    ModelRegistry,
    infer_capabilities_from_config,
)


@pytest.fixture
def sample_registry_file(temp_dir):
    registry_path = temp_dir / "Verify_models.yaml"
    registry_content = {
        "tiny-llama": {
            "repo_id": "fake-repo/tiny-llama",
        },
        "tiny-t5": {
            "repo_id": "fake-repo/tiny-t5",
        },
    }

    with registry_path.open("w") as f:
        yaml.dump(registry_content, f)

    return registry_path


@pytest.fixture
def mock_config_test(monkeypatch):
    def mock_from_pretrained(repo_id):
        if "llama" in repo_id.lower():
            config = PretrainedConfig()
            config.architectures = ["LlamaForCausalLM"]
            config.model_type = "llama"
            return config
        elif "t5" in repo_id.lower():
            config = PretrainedConfig()
            config.architectures = ["T5ForConditionalGeneration"]
            config.is_encoder_decoder = True
            config.model_type = "t5"
            return config
        raise ValueError(f"Unknown model: {repo_id}")

    monkeypatch.setattr(AutoConfig, "from_pretrained", mock_from_pretrained)


@pytest.mark.meta
class TestModelRegistry:
    def test_load_registry(self, sample_registry_file, mock_config_test):
        registry = ModelRegistry(sample_registry_file)

        assert len(registry._models) == 2
        assert "tiny-llama" in registry._models
        assert "tiny-t5" in registry._models

        llama = registry._models["tiny-llama"]
        assert llama.name == "tiny-llama"
        assert llama.repo_id == "fake-repo/tiny-llama"
        assert ModelCapability.CAUSAL_LM in llama.capabilities
        assert ModelCapability.SEQ2SEQ_LM not in llama.capabilities

        t5 = registry._models["tiny-t5"]
        assert t5.name == "tiny-t5"
        assert t5.repo_id == "fake-repo/tiny-t5"
        assert ModelCapability.SEQ2SEQ_LM in t5.capabilities

    @pytest.mark.parametrize(
        "capability,model_names,expected_models",
        [
            (ModelCapability.CAUSAL_LM, None, ["tiny-llama"]),
            (ModelCapability.SEQ2SEQ_LM, None, ["tiny-t5"]),
            (~ModelCapability.SEQ2SEQ_LM, None, ["tiny-llama"]),
            (ModelCapability.CAUSAL_LM | ModelCapability.SEQ2SEQ_LM, None, ["tiny-llama", "tiny-t5"]),
            (ModelCapability.CAUSAL_LM & ModelCapability.SEQ2SEQ_LM, None, []),
            (None, ["tiny-llama"], ["tiny-llama"]),
            (None, ["tiny-llama", "tiny-t5"], ["tiny-llama", "tiny-t5"]),
            (ModelCapability.CAUSAL_LM, "tiny-llama", ["tiny-llama"]),
            (ModelCapability.CAUSAL_LM, ["tiny-t5"], []),
        ],
    )
    def test_get_matching_models(
        self, sample_registry_file, mock_config_test, capability, model_names, expected_models
    ):
        registry = ModelRegistry(sample_registry_file)
        matches = registry.get_matching_models(capability, model_names=model_names)
        assert sorted([m.name for m in matches]) == sorted(expected_models)

    def test_get_random_matching_model_history(self, sample_registry_file, mock_config_test):
        registry = ModelRegistry(sample_registry_file)

        # First call should return first model (sorted by usage count)
        model1 = registry.get_random_matching_model()
        assert registry._random_history[model1.name] == 1

        # Second call should return different model
        model2 = registry.get_random_matching_model()
        assert model2.name != model1.name
        assert registry._random_history[model2.name] == 1

        # Third call should return model1 again as they're now equal
        model3 = registry.get_random_matching_model()
        assert model3.name == model1.name
        assert registry._random_history[model1.name] == 2

    def test_get_random_matching_model_no_matches(self, sample_registry_file, mock_config_test):
        registry = ModelRegistry(sample_registry_file)

        with pytest.raises(ValueError, match="No models match criteria"):
            registry.get_random_matching_model(model_names=["non-existent-model"])


@pytest.mark.meta
class TestCapabilityExpression:
    def test_simple_condition(self):
        expr = CapabilityExpression(lambda caps: ModelCapability.CAUSAL_LM in caps)
        assert expr.evaluate({ModelCapability.CAUSAL_LM})
        assert not expr.evaluate({ModelCapability.SEQ2SEQ_LM})

    def test_and_operator(self):
        expr1 = CapabilityExpression(lambda caps: ModelCapability.CAUSAL_LM in caps)
        expr2 = CapabilityExpression(lambda caps: ModelCapability.SEQ2SEQ_LM in caps)
        combined = expr1 & expr2

        assert not combined.evaluate({ModelCapability.CAUSAL_LM})
        assert not combined.evaluate({ModelCapability.SEQ2SEQ_LM})
        assert combined.evaluate({ModelCapability.CAUSAL_LM, ModelCapability.SEQ2SEQ_LM})

    def test_or_operator(self):
        expr1 = CapabilityExpression(lambda caps: ModelCapability.CAUSAL_LM in caps)
        expr2 = CapabilityExpression(lambda caps: ModelCapability.SEQ2SEQ_LM in caps)
        combined = expr1 | expr2

        assert combined.evaluate({ModelCapability.CAUSAL_LM})
        assert combined.evaluate({ModelCapability.SEQ2SEQ_LM})
        assert combined.evaluate({ModelCapability.CAUSAL_LM, ModelCapability.SEQ2SEQ_LM})
        assert not combined.evaluate(set())

    def test_not_operator(self):
        expr = CapabilityExpression(lambda caps: ModelCapability.CAUSAL_LM in caps)
        inverted = ~expr

        assert not inverted.evaluate({ModelCapability.CAUSAL_LM})
        assert inverted.evaluate({ModelCapability.SEQ2SEQ_LM})
        assert inverted.evaluate(set())


@pytest.mark.meta
class TestModelCapability:
    def test_from_str(self):
        assert ModelCapability.from_str("CAUSAL_LM") == ModelCapability.CAUSAL_LM
        assert ModelCapability.from_str("SEQ2SEQ_LM") == ModelCapability.SEQ2SEQ_LM

        with pytest.raises(KeyError):
            ModelCapability.from_str("INVALID")

    def test_to_expression(self):
        expr = ModelCapability.CAUSAL_LM.to_expression()
        assert isinstance(expr, CapabilityExpression)
        assert expr.evaluate({ModelCapability.CAUSAL_LM})
        assert not expr.evaluate({ModelCapability.SEQ2SEQ_LM})

    def test_capability_and(self):
        expr = ModelCapability.CAUSAL_LM & ModelCapability.SEQ2SEQ_LM
        assert isinstance(expr, CapabilityExpression)
        assert not expr.evaluate({ModelCapability.CAUSAL_LM})
        assert not expr.evaluate({ModelCapability.SEQ2SEQ_LM})
        assert expr.evaluate({ModelCapability.CAUSAL_LM, ModelCapability.SEQ2SEQ_LM})

    def test_capability_or(self):
        expr = ModelCapability.CAUSAL_LM | ModelCapability.SEQ2SEQ_LM
        assert isinstance(expr, CapabilityExpression)
        assert expr.evaluate({ModelCapability.CAUSAL_LM})
        assert expr.evaluate({ModelCapability.SEQ2SEQ_LM})
        assert expr.evaluate({ModelCapability.CAUSAL_LM, ModelCapability.SEQ2SEQ_LM})

    def test_capability_not(self):
        expr = ~ModelCapability.CAUSAL_LM
        assert isinstance(expr, CapabilityExpression)
        assert not expr.evaluate({ModelCapability.CAUSAL_LM})
        assert expr.evaluate({ModelCapability.SEQ2SEQ_LM})
        assert expr.evaluate(set())


@pytest.mark.meta
class TestInferCapabilities:
    @pytest.fixture
    def causal_lm_config(self):
        config = PretrainedConfig()
        config.architectures = ["LlamaForCausalLM"]
        config.model_type = "llama"
        config.is_encoder_decoder = False
        return config

    @pytest.fixture
    def seq2seq_config(self):
        config = PretrainedConfig()
        config.architectures = ["T5ForConditionalGeneration"]
        config.model_type = "t5"
        config.is_encoder_decoder = True
        return config

    def test_infer_causal_lm_from_architecture(self, causal_lm_config):
        capabilities = infer_capabilities_from_config(causal_lm_config)
        assert ModelCapability.CAUSAL_LM in capabilities
        assert ModelCapability.SEQ2SEQ_LM not in capabilities

    def test_infer_seq2seq_from_architecture(self, seq2seq_config):
        capabilities = infer_capabilities_from_config(seq2seq_config)
        assert ModelCapability.SEQ2SEQ_LM in capabilities
        assert ModelCapability.CAUSAL_LM not in capabilities

    def test_empty_config(self):
        config = PretrainedConfig()
        capabilities = infer_capabilities_from_config(config)
        assert len(capabilities) == 0
