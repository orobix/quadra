import functools
from collections.abc import Callable

from quadra.utils.utils import get_logger

logger = get_logger(__name__)


def deprecated(message: str) -> Callable:
    """Decorator to mark a function as deprecated.

    Args:
        message: Message to be displayed when the function is called.

    Returns:
        Decoratored function.
    """

    def deprecated_decorator(func_or_class: Callable) -> Callable:
        """Decorator to mark a function as deprecated."""

        @functools.wraps(func_or_class)
        def wrapper(*args, **kwargs):
            """Wrapper function to display a warning message."""
            warning_msg = f"{func_or_class.__name__} is deprecated. {message}"
            logger.warning(warning_msg)
            return func_or_class(*args, **kwargs)

        return wrapper

    return deprecated_decorator
