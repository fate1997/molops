#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

setup(
    author="Gaopeng Ren",
    description="Package containing useful operations to manipulate molecules.",
    name='molops',
    packages=find_packages(include=['molops', 'molops.*', 'molops.*.*']),
    include_package_data=True,
    version='0.0.1',
)