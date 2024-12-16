import inspect
from dataclasses import field, fields, make_dataclass
from functools import wraps
from typing import Any, Protocol, TypeVar, cast, get_type_hints

from peft import LoraConfig


DEFAULT_SEP_SEQ = "\n\n"


class DataclassProtocol(Protocol):
    """Protocol defining the required interface for dataclass operations.

    Attributes:
        __dataclass_fields__: Dictionary mapping field names to field objects
        __name__: Name of the dataclass
        __dataclass_params__: Dataclass configuration parameters
    """

    __dataclass_fields__: dict[str, Any]
    __name__: str
    __dataclass_params__: Any


T = TypeVar("T", bound=DataclassProtocol)


def suffix_dataclass_factory(base_class: type[T], suffix: str = "_A") -> type[T]:
    """Creates a new dataclass by appending a suffix to the base class name and all its fields.

    This function creates a new dataclass that mirrors the structure of a base dataclass,
    but with modified field names. It's useful for creating parallel configurations where
    you need similar but distinct settings for different components.

    Args:
        base_class (type[T]): Base dataclass to derive from
        suffix (str, optional): Suffix to append to the base class name and fields. Defaults to "_A"

    Returns:
        type[T]: New dataclass with suffixed name and fields

    Raises:
        TypeError: If base_class is not a dataclass

    Examples:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class BaseConfig:
        ...     learning_rate: float = 0.001
        ...     batch_size: int = 32
        >>>
        >>> ConfigA = suffix_dataclass_factory(BaseConfig, "_A")
        >>> config_a = ConfigA(learning_rate_A=0.002, batch_size_A=64)
        >>> config_a
        ConfigA(learning_rate_A=0.002, batch_size_A=64)

    Notes:
        - The new dataclass is created without inheriting from the base class
        - All field attributes (init, repr, compare, etc.) are preserved
        - Default values and factory functions are maintained
    """
    if not hasattr(base_class, "__dataclass_fields__"):
        raise TypeError("Base class must be a dataclass")

    original_fields = fields(cast(Any, base_class))
    type_hints = get_type_hints(base_class)

    new_fields = []
    for og_field in original_fields:
        field_type = type_hints.get(og_field.name, Any)
        new_field_name = og_field.name + suffix

        # Create field with explicit parameters instead of dict
        field_obj = field(
            init=bool(og_field.init),
            repr=bool(og_field.repr),
            compare=bool(og_field.compare),
            metadata=dict(og_field.metadata or {}),
        )

        # Handle default values
        if og_field.default is not og_field.default_factory:
            field_obj.default = og_field.default
        if og_field.default_factory is not og_field.default_factory.__class__:
            field_obj.default_factory = og_field.default_factory

        new_fields.append((new_field_name, field_type, field_obj))

    # Create new dataclass without inheriting from base class
    new_class_name = base_class.__name__ + suffix.replace("_", "")
    new_class = make_dataclass(
        new_class_name,
        new_fields,
        bases=(),  # No inheritance
        frozen=getattr(base_class, "__dataclass_params__").frozen,
    )
    return new_class


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
