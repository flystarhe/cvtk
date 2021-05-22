from setuptools import setup, find_packages
from pathlib import Path
import re


here = Path(__file__).parent.resolve()


with open(here / "cvtk/__init__.py", "r") as fh:
    version = re.search(r"__version__ = \"(.*?)\"", fh.read()).group(1)


long_description = (here / "README.md").read_text(encoding="utf-8")


def parse_requirements(fpath):

    def gen_packages_items():
        with open(fpath, "r") as fh:
            for line in fh.readlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    yield line

    packages = list(gen_packages_items())
    return packages


install_requires = parse_requirements(here / "requirements.txt")


setup(
    name="cvtk",
    version=version,
    author="flystarhe",
    author_email="flystarhe@qq.com",
    keywords="computer vision, machine learning",
    description="computer vision toolkit of pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/flystarhe/cvtk",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    packages=find_packages(),
    python_requires=">=3.6, <4",
    install_requires=install_requires,
)
