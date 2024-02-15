"Setup script for package"
import pathlib
from setuptools import setup

short_description = """Practical part of my bachelor thesis
    on Multiple-criteria Decision Making."""

setup(
    name="mymcdm",
    version="1.1.1",
    description=short_description,
    author="Marek Brodacký",
    author_email="brodackym@gmail.com",
    keywords="mcdm, madm, decision-making, normalization methods",
    ulr="https://github.com/slunimara",
    install_requires=["numpy>=1.24.2", "pandas>=1.5.3"],
    entry_points={
        "console_scripts": [
            "mymcdm = mymcdm.cli:cli",
        ]
    },
    long_description=pathlib.Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Typing :: Typed",
    ],
)
