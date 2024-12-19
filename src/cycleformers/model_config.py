from dataclasses import dataclass
from typing import Literal

from .utils import prefixed_view


@dataclass
class ModelConfig:
    """https://github.com/huggingface/trl/blob/main/trl/trainer/model_config.py

    Parameters:
        model_name_or_path (`Optional[str]`, *optional*, defaults to `None`):
            Model checkpoint for weights initialization.
        model_revision (`str`, *optional*, defaults to `"main"`):
            Specific model version to use. It can be a branch name, a tag name, or a commit id.
        torch_dtype (`Optional[Literal["auto", "bfloat16", "float16", "float32"]]`, *optional*, defaults to `None`):
            Override the default `torch.dtype` and load the model under this dtype. Possible values are

                - `"bfloat16"`: `torch.bfloat16`
                - `"float16"`: `torch.float16`
                - `"float32"`: `torch.float32`
                - `"auto"`: Automatically derive the dtype from the model's weights.

        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether to allow for custom models defined on the Hub in their own modeling files. This option should only
            be set to `True` for repositories you trust and in which you have read the code, as it will execute code
            present on the Hub on your local machine.
        attn_implementation (`Optional[str]`, *optional*, defaults to `None`):
            Which attention implementation to use. You can run `--attn_implementation=flash_attention_2`, in which case
            you must install this manually by running `pip install flash-attn --no-build-isolation`.
        use_peft (`bool`, *optional*, defaults to `False`):
            Whether to use PEFT for training.
        lora_r (`int`, *optional*, defaults to `16`):
            LoRA R value.
        lora_alpha (`int`, *optional*, defaults to `32`):
            LoRA alpha.
        lora_dropout (`float`, *optional*, defaults to `0.05`):
            LoRA dropout.
        lora_target_modules (`Optional[Union[str, list[str]]]`, *optional*, defaults to `None`):
            LoRA target modules.
        lora_modules_to_save (`Optional[list[str]]`, *optional*, defaults to `None`):
            Model layers to unfreeze & train.
        lora_task_type (`str`, *optional*, defaults to `"CAUSAL_LM"`):
            Task type to pass for LoRA (use `"SEQ_CLS"` for reward modeling).
        use_rslora (`bool`, *optional*, defaults to `False`):
            Whether to use Rank-Stabilized LoRA, which sets the adapter scaling factor to `lora_alpha/√r`, instead of
            the original default value of `lora_alpha/r`."""

    model_name_or_path: str | None = None
    model_revision: str = "main"
    torch_dtype: Literal["auto", "bfloat16", "float16", "float32"] | None = None
    trust_remote_code: bool = False
    attn_implementation: str | None = None
    use_peft: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None
    lora_modules_to_save: list[str] | None = None
    lora_task_type: str = "CAUSAL_LM"
    use_rslora: bool = False
    use_dora: bool = False

    def __post_init__(self):
        self._A = None
        self._B = None

    @property
    def A(self) -> "ModelConfig":
        return self._A

    @A.setter
    def A(self, value: "ModelConfig"):
        self._A = value

    @property
    def B(self) -> "ModelConfig":
        return self._B

    @B.setter
    def B(self, value: "ModelConfig"):
        self._B = value


@dataclass
@prefixed_view(ModelConfig, "A_")
class ModelConfigA:
    pass


@dataclass
@prefixed_view(ModelConfig, "B_")
class ModelConfigB:
    pass


__all__ = ["ModelConfig", "ModelConfigA", "ModelConfigB"]
