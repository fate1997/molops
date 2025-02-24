import pathlib

import pytest


def pytest_configure():
    this_dir = pathlib.Path(__file__).parent
    pytest.EXAMPLE_PATH = this_dir.joinpath("example")