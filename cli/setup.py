#!/usr/bin/env python3
"""
Setup script for Call Upload CLI
"""

from setuptools import setup, find_packages

setup(
    name="call-cli",
    version="1.0.0",
    description="Upload call recordings, monitor transcription, and export transcripts",
    author="Call Transcription Team",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "click>=8.0.0",
        "rich>=13.0.0",
        "snowflake-connector-python>=3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "call-cli=cli.call_cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
)
