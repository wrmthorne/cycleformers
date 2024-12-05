from functools import wraps
import inspect


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
