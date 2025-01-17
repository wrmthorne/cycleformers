from collections import Counter
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Union

import yaml
from transformers import AutoConfig, PretrainedConfig


class CapabilityExpression:
    """Expression for filtering models based on their capabilities."""

    def __init__(self, condition: Callable[[set["ModelCapability"]], bool]):
        self.condition = condition

    def evaluate(self, capabilities: set["ModelCapability"]) -> bool:
        return self.condition(capabilities)

    def __and__(self, other: "CapabilityExpression") -> "CapabilityExpression":
        return CapabilityExpression(lambda caps: self.condition(caps) and other.condition(caps))

    def __or__(self, other: "CapabilityExpression") -> "CapabilityExpression":
        return CapabilityExpression(lambda caps: self.condition(caps) or other.condition(caps))

    def __invert__(self) -> "CapabilityExpression":
        return CapabilityExpression(lambda caps: not self.condition(caps))


class ModelCapability(Enum):
    SEQ2SEQ_LM = auto()
    CAUSAL_LM = auto()

    @classmethod
    def from_str(cls, name: str) -> "ModelCapability":
        return cls[name]

    def to_expression(self) -> "CapabilityExpression":
        return CapabilityExpression(lambda caps: self in caps)

    def __and__(self, other: Union["ModelCapability", "CapabilityExpression"]) -> "CapabilityExpression":
        return self.to_expression() & (other.to_expression() if isinstance(other, ModelCapability) else other)

    def __or__(self, other: Union["ModelCapability", "CapabilityExpression"]) -> "CapabilityExpression":
        return self.to_expression() | (other.to_expression() if isinstance(other, ModelCapability) else other)

    def __invert__(self) -> "CapabilityExpression":
        return ~self.to_expression()


def infer_capabilities_from_config(config: PretrainedConfig) -> set[ModelCapability]:
    """Infer model capabilities from HuggingFace config.

    Args:
        config: HuggingFace model config

    Returns:
        Set of model capabilities inferred from the config.
    """
    capabilities = set()

    # Architecture-based capabilities
    if hasattr(config, "is_encoder_decoder") and config.is_encoder_decoder:
        capabilities.add(ModelCapability.SEQ2SEQ_LM)

    if hasattr(config, "architectures") and config.architectures:
        if any("ForCausalLM" in arch for arch in config.architectures):
            capabilities.add(ModelCapability.CAUSAL_LM)

    return capabilities


@dataclass
class ModelSpec:
    name: str
    repo_id: str
    capabilities: set[ModelCapability]
    config: PretrainedConfig
    description: str = ""  # Any notes that are important to remember about the model

    def __hash__(self) -> int:
        return hash((self.name, self.repo_id))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelSpec):
            return NotImplemented
        return self.name == other.name and self.repo_id == other.repo_id

    @classmethod
    def from_hub(cls, name: str, repo_id: str) -> "ModelSpec":
        config = AutoConfig.from_pretrained(repo_id)
        capabilities = infer_capabilities_from_config(config)
        return cls(name=name, repo_id=repo_id, capabilities=capabilities, config=config)


class ModelRegistry:
    def __init__(self, registry_path: Path):
        self._models: dict[str, ModelSpec] = {}
        self.load_registry(registry_path)

    def load_registry(self, registry_path: Path):
        with registry_path.open() as f:
            registry_dict = yaml.safe_load(f)

        for name, spec in registry_dict.items():
            self._models[name] = ModelSpec.from_hub(name=name, repo_id=spec["repo_id"])

    def get_matching_models(
        self,
        capability_expr: ModelCapability | CapabilityExpression | None = None,
        model_names: list[str] | str | None = None,
    ) -> list[ModelSpec]:
        """Get models matching capability expression AND model names.

        Args:
            capability_expr: Capability expression to match. Can be a ModelCapability or a CapabilityExpression.
                If None, no capability filtering is applied.
            model_names: List of model names to match. Can be a single string or a list of strings.
                If None, no name filtering is applied.

        Returns:
            List of models matching both the capability expression and model names.
            If both capability_expr and model_names are None, returns all models.

        Examples:
            >>> registry = ModelRegistry(Path("models_to_test.yaml"))
            >>> # Find models that are NOT seq2seq
            >>> models = registry.get_matching_models(~ModelCapability.SEQ2SEQ_LM)

            >>> # Find models that are causal LM
            >>> models = registry.get_matching_models(ModelCapability.CAUSAL_LM)

            >>> # Find models that are either causal LM or seq2seq
            >>> models = registry.get_matching_models(
            ...     ModelCapability.CAUSAL_LM | ModelCapability.SEQ2SEQ_LM
            ... )

            >>> # Find models that are both causal LM and seq2seq (empty list)
            >>> models = registry.get_matching_models(
            ...     ModelCapability.CAUSAL_LM & ModelCapability.SEQ2SEQ_LM
            ... )

            >>> # Get specific models by name
            >>> models = registry.get_matching_models(model_names=["tiny-llama", "tiny-t5"])

            >>> # Combine capability and name filters
            >>> models = registry.get_matching_models(
            ...     capability_expr=ModelCapability.SEQ2SEQ_LM,
            ...     model_names=["tiny-t5"]
            ... )

            >>> # Return tiny-llama-3.1
            >>> models = registry.get_matching_models(model_names="tiny-llama-3.1")

            >>> # Get all models
            >>> models = registry.get_matching_models()
        """
        if capability_expr is None and model_names is None:
            return list(self._models.values())

        if isinstance(capability_expr, ModelCapability):
            capability_expr = capability_expr.to_expression()

        if isinstance(model_names, str):
            model_names = [model_names]

        matches = []
        for spec in self._models.values():
            if capability_expr is not None:
                if not capability_expr.evaluate(spec.capabilities):
                    continue

            if model_names is not None:
                if spec.name not in model_names:
                    continue

            matches.append(spec)

        return matches
