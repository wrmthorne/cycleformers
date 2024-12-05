from functools import wraps
import inspect
from dataclasses import dataclass, fields, make_dataclass, field
from typing import get_type_hints, Any


def suffix_dataclass_factory(base_class: dataclass, suffix: str = '_A') -> dataclass:
    """
    Creates a new dataclass by appending a suffix to the base class name and all its fields.

    args:
        base_class: Base dataclass to derive from
        suffix: Suffix to append to the base class name and all its fields

    returns:
        New dataclass with suffixed name and fields
    """
    if not hasattr(base_class, '__dataclass_fields__'):
        raise TypeError("Base class must be a dataclass")
    
    original_fields = fields(base_class)
    type_hints = get_type_hints(base_class)

    new_fields = []
    for og_field in original_fields:
        field_type = type_hints.get(og_field.name, Any)
        new_field_name = og_field.name + suffix
        
        # Preserve metadata
        field_kwargs = {
            'init': og_field.init,
            'repr': og_field.repr,
            'compare': og_field.compare,
            'metadata': og_field.metadata,
        }
        
        # Handle default values
        if og_field.default is not og_field.default_factory:
            field_kwargs['default'] = og_field.default
        if og_field.default_factory is not og_field.default_factory:
            field_kwargs['default_factory'] = og_field.default_factory
            
        new_fields.append((new_field_name, field_type, field(**field_kwargs)))
    
    # Create new dataclass without inheriting from base class
    new_class_name = base_class.__name__ + suffix.replace('_', '')
    new_class = make_dataclass(
        new_class_name,
        new_fields,
        bases=(),  # No inheritance
        frozen=getattr(base_class, '__dataclass_params__').frozen
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
                if attr in bound_args.arguments and attr != 'self':
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
