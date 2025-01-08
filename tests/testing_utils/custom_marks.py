import pytest

from cycleformers.import_utils import is_liger_kernel_available, is_peft_available


def requires_peft(func):
    """Decorator to skip test if PEFT is not available"""
    return pytest.mark.skipif(not is_peft_available(), reason="PEFT not available")(func)


def requires_liger_kernel(func):
    """Decorator to skip test if Liger kernel is not available"""
    return pytest.mark.skipif(not is_liger_kernel_available(), reason="Liger kernel not available")(func)
