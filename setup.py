from setuptools import setup, find_packages
from pathlib import Path


here = Path(__file__).parent.resolve()
version = (here / "version.txt").read_text(encoding="utf-8")
long_description = (here / "readme.md").read_text(encoding="utf-8")
install_requires = (here / "requirements.txt").read_text(encoding="utf-8").split()


setup(
    name="cvtk",
    version=version.strip(),
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
