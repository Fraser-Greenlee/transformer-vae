import os
import re
from setuptools import setup

package = "transformer_vae"
dirname = os.path.dirname(__file__)


def file_to_string(*path):
    with open(os.path.join(dirname, *path), encoding="utf8") as f:
        return f.read()


# can't import __version__ so extract it manually
contents = file_to_string(package, "__init__.py")
__version__ = re.search(r'__version__ = "([.\d]+)"', contents).group(1)

install_requires = [
    "datasets[s3]>=1.5.0",
    'sagemaker>=2.31.0'
    "transformers",
    "wandb",
    "torch==1.8.0",
    "sklearn",  # for SVM
    "torch-dct"
]

tests_require = ["pytest", "flake8", "flake8-mypy", "black", "twine"]

setup(
    name=package,
    version=__version__,
    description="Interpolate between discrete sequences.",
    long_description=file_to_string("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/Fraser-Greenlee/transformer-vae",
    author="Fraser Greenlee",
    author_email="fraser.greenlee@mac.com",
    license="MIT",
    packages=[package],
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require={
        "test": tests_require,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
