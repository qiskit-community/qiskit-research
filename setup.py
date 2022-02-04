from setuptools import find_packages, setup

name = "qiskit-research"
version = "0.0.1"
description = (
    "Research using Qiskit, demonstrating best practices "
    "for running quantum computing experiments."
)

with open("README.md") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    install_requires = f.read()

setup(
    name=name,
    version=version,
    description=description,
    long_description=long_description,
    install_requires=install_requires,
    packages=find_packages(),
)
