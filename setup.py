#!/usr/bin/env python3
"""Setup script for MacrOSINT package."""
from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Read requirements
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="MacrOSINT",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Commodities Dashboard - Multi-page Dash application for analyzing agricultural and energy commodity data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/macrOS-Int",
    packages=find_packages(include=['MacrOSINT', 'MacrOSINT.*']),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-asyncio>=0.21.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
        ],
        'download': [
            'gdown>=4.7.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'macrosint-setup=MacrOSINT.setup_utils:main',
        ],
    },
    include_package_data=True,
    package_data={
        'MacrOSINT': [
            '*.toml',
            'data/sources/*/*.toml',
            'components/plotting/*.toml',
            'assets/*',
        ],
    },
    zip_safe=False,
)
