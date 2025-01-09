import time
from functools import wraps

import torch


def record_function_wrapper(name=None):
    """Decorator to profile a function's execution time and memory usage.

    Args:
        name (str, optional): Custom name for the profiling label. Defaults to function name.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            step_start_time = time.perf_counter()

            # Clear cache before step if CUDA is available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            profile_name = name or f"## {func.__name__} ##"
            with torch.profiler.record_function(profile_name):
                result = func(self, *args, **kwargs)

            # Track timing and memory if the object has profiling_stats
            if hasattr(self, "profiling_stats"):
                step_duration = time.perf_counter() - step_start_time
                self.profiling_stats["step_times"].append(step_duration)

                # Log GPU memory
                if hasattr(self, "_log_gpu_memory"):
                    current_memory, max_memory = self._log_gpu_memory() or (0, 0)

                    # Update metrics if result is a dict
                    if isinstance(result, dict):
                        result.update(
                            {
                                "gpu_memory_used": current_memory,
                                "gpu_memory_max": max_memory,
                            }
                        )

            return result

        return wrapper

    return decorator
