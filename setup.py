from setuptools import setup, find_packages

setup(
    name="CosmoTree",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "numba",
    ],
    author="Jules",
    description="A Numba-accelerated tree builder for cosmological correlation functions.",
)
