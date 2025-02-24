import os
from contextlib import contextmanager


@contextmanager
def run_in(workdir: str):
    """Context manager to run a block of code in a specified directory."""
    original_dir = os.path.abspath(os.getcwd())
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    try:
        yield
    finally:
        os.chdir(original_dir)

def at_workdir(path: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with run_in(path):
                return func(*args, **kwargs)
        return wrapper
    return decorator
