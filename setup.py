#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

setup_requirements = [
    "pytest-runner>=5.2",
]

plot_requirements = [
    "matplotlib>=3.5.1",
    "seaborn>=0.11.2",
    # Support
    "pyarrow>=7.0.0",
    # Widgets
    "ipywidgets>=7.7.0",
]

test_requirements = [
    *plot_requirements,
    "black>=22.3.0",
    "codecov>=2.1.4",
    "flake8>=3.8.3",
    "flake8-debugger>=3.2.1",
    "isort>=5.7.0",
    "mypy>=0.790",
    "pytest>=5.4.3",
    "pytest-cov>=2.9.0",
    "pytest-raises>=0.11",
    "tox>=3.15.2",
]

dev_requirements = [
    *setup_requirements,
    *test_requirements,
    "bump2version>=1.0.1",
    "coverage>=5.1",
    "jupyterlab>=3.2.8",
    "m2r2>=0.2.7",
    "pytest-runner>=5.2",
    "Sphinx>=3.4.3",
    "furo>=2022.4.7",
    "twine>=3.1.1",
    "wheel>=0.34.2",
]

requirements = [
    "cdp-backend>=3.0.3",
    "nltk>=3.6",
    "numpy>=1.22.3",
    "pandas>=1.4.1",
    "tqdm>=4.63.1",
    # Version pins set by cdp-backend
    "dataclasses_json",
    "fireo",
    "gcsfs",
]

extra_requirements = {
    "setup": setup_requirements,
    "test": test_requirements,
    "dev": dev_requirements,
    "plot": plot_requirements,
    "all": [
        *requirements,
        *dev_requirements,
        *plot_requirements,
    ],
}

setup(
    author="Council Data Project Contributors",
    author_email="jmaxfieldbrown@gmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description=(
        "Data Utilities and Processing Functions for Generalized for all CDP Instances"
    ),
    entry_points={
        "console_scripts": [
            (
                "generate_cdp_councils_in_action_2022_paper_content="
                "cdp_data.bin.research.cdp_councils_in_action_2022_content:main"
            ),
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="Council Data Project, CDP, NLP, Natural Language Processing",
    name="cdp-data",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    python_requires=">=3.8",
    setup_requires=setup_requirements,
    test_suite="cdp_data/tests",
    tests_require=test_requirements,
    extras_require=extra_requirements,
    url="https://github.com/CouncilDataProject/cdp-data",
    # Do not edit this string manually, always use bumpversion
    # Details in CONTRIBUTING.rst
    version="0.0.4",
    zip_safe=False,
)
