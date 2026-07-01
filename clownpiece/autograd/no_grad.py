from contextlib import contextmanager
import threading

# Thread-local storage for gradient computation state.
# Each thread maintains its own _grad_enabled flag, so that one thread's
# no_grad() context does not leak into other threads (fixes "roots is empty"
# error in multi-threaded backward).
_local = threading.local()

def _get_local():
    """Return this thread's local storage, initializing grad_enabled if needed."""
    if not hasattr(_local, 'grad_enabled'):
        _local.grad_enabled = True
    return _local

def is_grad_enabled():
    """Returns whether gradient tracking is currently enabled (for this thread)."""
    return _get_local().grad_enabled

@contextmanager
def no_grad():
    """
    Context-manager that disables gradient calculation.

    Within this context, gradients will not be calculated, and `requires_grad` flags
    will be ignored. This can be used to improve performance when you don't need
    gradients, such as during inference.

    Example:
        ```python
        with no_grad():
            # Computations here don't track gradients
            result = model(input_data)
        ```
    """
    local = _get_local()
    previous = local.grad_enabled
    local.grad_enabled = False
    try:
        yield
    finally:
        local.grad_enabled = previous

class set_grad_enabled:
    """
    Context-manager that sets gradient calculation to on or off.

    Parameters:
        mode (bool): Flag whether to enable gradients (True) or disable (False)

    Example:
        ```python
        with set_grad_enabled(False):
            # Computations here don't track gradients
            result = model(input_data)
        ```
    """
    def __init__(self, mode):
        self.mode = mode
        self.prev = is_grad_enabled()

    def __enter__(self):
        _get_local().grad_enabled = self.mode

    def __exit__(self, exc_type, exc_val, exc_tb):
        _get_local().grad_enabled = self.prev
