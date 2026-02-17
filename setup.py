from setuptools import setup, Extension

setup(
    name="arithmetic_coder",
    ext_modules=[
        Extension(
            "arithmetic_coder",
            sources=["arith_coder.c"],
            extra_compile_args=["-O3", "-march=native"],
        )
    ],
)
