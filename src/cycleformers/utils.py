import inspect
from dataclasses import MISSING, fields
from functools import wraps
from typing import get_type_hints

from peft import LoraConfig
from transformers.hf_argparser import DataClassType

from .import_utils import is_liger_kernel_available


if is_liger_kernel_available():
    from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN  # type: ignore

    VALID_LIGER_MODELS = MODEL_TYPE_TO_APPLY_LIGER_FN.keys()
else:
    VALID_LIGER_MODELS = []

DEFAULT_SEP_SEQ = "\n\n"


def prefixed_view(base_class: DataClassType, prefix: str):
    """Creates a dataclass-like decorator that provides a prefixed view of another dataclass.
    When instantiated, returns an instance of the original dataclass with unprefixed attributes.

    This decorator allows you to create a class that acts as a view of another dataclass,
    where all attributes are prefixed. When instantiating the prefixed class, it returns
    an instance of the original dataclass with the prefixes removed.

    Args:
        base_class: The original dataclass to create a view of
        prefix: The prefix to add to all attribute names

    Returns:
        A decorator function that creates the prefixed view class

    Example:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Config:
        ...     name: str
        ...     value: int = 42
        ...
        >>> @prefixed_view(Config, "test_")
        ... class TestConfig:
        ...     pass
        ...
        >>> # The new class has prefixed type hints
        >>> TestConfig.__annotations__
        {'test_name': <class 'str'>, 'test_value': <class 'int'>}
        >>>
        >>> # Creating an instance with prefixed attributes
        >>> config = TestConfig(test_name="example", test_value=100)
        >>> config
        Config(name='example', value=100)
        >>>
        >>> # Invalid attributes raise TypeError
        >>> config = TestConfig(invalid_attr="test")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        TypeError: Unexpected argument: invalid_attr
        >>>
        >>> # Default values are preserved
        >>> config = TestConfig(test_name="example")
        >>> config
        Config(name='example', value=42)
    """

    def wrapper(cls):
        # Get original class type hints and fields
        base_hints = get_type_hints(base_class)
        base_fields = {f.name: f for f in fields(base_class)}

        # Create new fields with prefixed names but same types/defaults
        new_annotations = {}
        new_defaults = {}

        for name, type_hint in base_hints.items():
            prefixed_name = f"{prefix}{name}"
            new_annotations[prefixed_name] = type_hint

            # Copy default values if they exist
            if name in base_fields:
                field = base_fields[name]
                if field.default is not MISSING:
                    new_defaults[prefixed_name] = field.default
                if field.default_factory is not MISSING:
                    new_defaults[prefixed_name] = field.default_factory()

        # Add annotations and default values to the class
        cls.__annotations__ = new_annotations
        for name, value in new_defaults.items():
            setattr(cls, name, value)

        def __new__(cls, **kwargs):
            # Validate input against our prefixed fields
            for key in kwargs:
                if key not in new_annotations:
                    raise TypeError(f"Unexpected argument: {key}")

            # Create mapping of unprefixed attributes
            unprefixed_kwargs = {key[len(prefix) :]: value for key, value in kwargs.items()}

            # Return instance of original dataclass
            return base_class(**unprefixed_kwargs)

        cls.__new__ = staticmethod(__new__)
        return cls

    return wrapper


def auto_temp_attributes(*attrs_to_cleanup):
    """Decorator that automatically manages temporary attributes on a class instance.

    This decorator solves the issue of methods that need to temporarily modify class attributes
    that might be needed by other methods. It automatically sets attributes based on method
    parameters and restores their original values (or removes them) after the method completes.

    Args:
        *attrs_to_cleanup: Variable number of attribute names to manage

    Returns:
        Callable: Decorated function that handles attribute lifecycle

    Examples:
        >>> class MyClass:
        ...     def __init__(self):
        ...         self.permanent = "permanent"
        ...
        ...     @auto_temp_attributes("model", "optimizer")
        ...     def my_method(self, model, optimizer=None):
        ...         print(f"Using {model} and {optimizer}")
        ...         print(f"Permanent: {self.permanent}")
        >>>
        >>> obj = MyClass()
        >>> obj.my_method("bert", optimizer="adam")
        Using bert and adam
        Permanent: permanent
        >>> hasattr(obj, "model")  # Attribute is cleaned up
        False

    Notes:
        - Original attribute values are restored after method execution
        - Attributes that didn't exist are removed
        - Works with both positional and keyword arguments
        - Handles exceptions by ensuring cleanup
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Store original attribute values
            original_values = {}
            for attr in attrs_to_cleanup:
                if hasattr(self, attr):
                    original_values[attr] = getattr(self, attr)

            # Get function signature to match positional args to parameter names
            sig = inspect.signature(func)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()

            # Set attributes based on parameters
            for attr in attrs_to_cleanup:
                # Skip 'self' parameter
                if attr in bound_args.arguments and attr != "self":
                    setattr(self, attr, bound_args.arguments[attr])
                else:
                    setattr(self, attr, None)

            try:
                # Execute the method
                result = func(self, *args, **kwargs)
                return result
            finally:
                # Clean up attributes
                for attr in attrs_to_cleanup:
                    if attr in original_values:
                        setattr(self, attr, original_values[attr])
                    else:
                        try:
                            delattr(self, attr)
                        except AttributeError:
                            pass

        return wrapper

    return decorator


def get_peft_config(model_config):
    """Creates a PEFT LoRA configuration from a model configuration dataclass."""
    if model_config.use_peft is False:
        return None

    peft_config = LoraConfig(
        task_type=model_config.lora_task_type,
        r=model_config.lora_r,
        target_modules=model_config.lora_target_modules,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        bias="none",
        use_rslora=model_config.use_rslora,
        modules_to_save=model_config.lora_modules_to_save,
    )

    return peft_config
