import os
import re
from setuptools import setup

package = "t5_vae"
dirname = os.path.dirname(__file__)


def file_to_string(*path):
    with open(os.path.join(dirname, *path), encoding="utf8") as f:
        return f.read()


# can't import __version__ so extract it manually
contents = file_to_string(package, "__init__.py")
__version__ = re.search(r'__version__ = "([.\d]+)"', contents).group(1)

install_requires = [
    "transformers==3.5.1",
    "datasets==1.1.2",
    "wandb==0.10.10",
    "torch==1.7.0",
]

tests_require = ["pytest", "flake8", "flake8-mypy", "black"]

# TODO allow executing training directly

setup(
    name=package,
    version=__version__,
    description="Interpolate between discrete sequences.",
    long_description=file_to_string("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/Fraser-Greenlee/T5-VAE",
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