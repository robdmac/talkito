#!/usr/bin/env python3

# TalkiTo - Universal TTS wrapper that works with any command
# Copyright (C) 2025 Robert Macrae
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
from setuptools import setup, find_packages

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read version from __version__.py
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), "talkito", "__version__.py")
    version_locals = {}
    with open(version_file, "r", encoding="utf-8") as f:
        exec(compile(f.read(), version_file, "exec"), version_locals)
    return version_locals["__version__"]

# Core dependencies (full feature set)
install_requires = [
    "SpeechRecognition>=3.8.1",
    "pyaudio>=0.2.11",
    "openai>=1.0.0",
    "boto3>=1.26.0",
    "amazon-transcribe>=0.6.0",
    "azure-cognitiveservices-speech>=1.24.0",
    "google-cloud-texttospeech>=2.14.0",
    "google-cloud-speech>=2.20.0",
    "elevenlabs>=0.2.0",
    "assemblyai>=0.5.0",
    "deepgram-sdk>=2.0.0",
    "twilio>=8.0.0",
    "slack-sdk>=3.19.0",
    "flask>=2.0.0",
    "waitress>=2.0.0",
    "kokoro>=0.9.4",
    "soundfile>=0.12.0",
    "numpy<2.0",
    "fastmcp>=0.1.0",
]

setup(
    name="talkito",
    version=get_version(),
    author="Robert Macrae",
    description="Universal TTS wrapper and voice interaction library for command-line programs",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/robdmac/talkito",
    project_urls={
        "Bug Tracker": "https://github.com/robdmac/talkito/issues",
        "Documentation": "https://github.com/robdmac/talkito/blob/main/README.md",
        "Source Code": "https://github.com/robdmac/talkito",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: System :: Shells",
        "Topic :: Utilities",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Environment :: Console",
    ],
    python_requires=">=3.10",
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "talkito=talkito.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "talkito": ["*.py"],
    },
    keywords=[
        "tts",
        "text-to-speech",
        "asr",
        "speech-recognition",
        "voice",
        "terminal",
        "cli",
        "command-line",
        "accessibility",
        "ai",
        "llm",
        "claude",
        "chatgpt",
        "voice-assistant",
    ],
)
