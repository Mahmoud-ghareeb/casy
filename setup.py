from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("memory.pyx"),
    zip_safe=False
)
