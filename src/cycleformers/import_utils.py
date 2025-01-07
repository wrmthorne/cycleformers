from transformers.utils.import_utils import _is_package_available


_liger_kernel_available = _is_package_available("liger-kernel")
_flash_attn_available = _is_package_available("flash-attn")
_peft_available = _is_package_available("peft")
_wandb_available = _is_package_available("wandb")


def is_liger_kernel_available() -> bool:
    return _liger_kernel_available


def is_flash_attn_available() -> bool:
    return _flash_attn_available


def is_peft_available() -> bool:
    return _peft_available


def is_wandb_available() -> bool:
    return _wandb_available
