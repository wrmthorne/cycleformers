import inspect
from dataclasses import field, fields, make_dataclass
from functools import wraps
from typing import Any, Protocol, TypeVar, cast, get_type_hints

from peft import LoraConfig


DEFAULT_SEP_SEQ = "\n\n"


class DataclassProtocol(Protocol):
    __dataclass_fields__: dict[str, Any]
    __name__: str
    __dataclass_params__: Any


T = TypeVar("T", bound=DataclassProtocol)


def suffix_dataclass_factory(base_class: type[T], suffix: str = "_A") -> type[T]:
    """
    Creates a new dataclass by appending a suffix to the base class name and all its fields.

    args:
        base_class: Base dataclass to derive from
        suffix: Suffix to append to the base class name and all its fields

    returns:
        New dataclass with suffixed name and fields
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
    """
    Decorator that automatically sets and manages temporary attributes on a class instance.
    This solves the issue of methods that modify attributes that are needed for other methods.
    Parameters matching attribute names are set to their passed values, others to None.
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
    """Helper function to return a PEFT config from a model config dataclass."""
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
