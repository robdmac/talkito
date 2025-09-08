#!/usr/bin/env python3

# Talkito - Universal TTS wrapper that works with any command
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

"""Utilities for model downloads with user consent and progress."""

import os
import sys
from typing import Callable
from pathlib import Path


def ask_user_consent(provider: str, model_name: str) -> bool:
    """Ask user for consent to download a model."""
    # Auto-approve if environment variable is set
    if os.environ.get('TALKITO_AUTO_APPROVE_DOWNLOADS', '').lower() in ('1', 'true', 'yes'):
        return True
    
    # Auto-decline in non-interactive environments to prevent hanging
    if not sys.stdin.isatty():
        print(f"Non-interactive environment detected. Declining download of {provider} model '{model_name}'.")
        print("Set TALKITO_AUTO_APPROVE_DOWNLOADS=1 to enable automatic downloads.")
        return False
    
    try:
        response = input(f"Download {provider} model '{model_name}'? (y/N): ").strip().lower()
        return response in ('y', 'yes')
    except (KeyboardInterrupt, EOFError):
        return False


def show_download_progress(provider: str, model_name: str):
    """Show simple download progress."""
    print(f"Downloading {provider} model '{model_name}'...")


def check_spacy_model_consent(provider: str, spacy_model: str = "en_core_web_sm") -> bool:
    """Check if spaCy language model needs consent and handle download."""
    try:
        import spacy
        # Try to load the model without downloading
        try:
            spacy.load(spacy_model, disable=[])
            return True  # Model already available
        except OSError:
            # Model not available, ask for consent
            if not ask_user_consent(f"{provider} (spaCy dependency)", f"language model '{spacy_model}'"):
                return False
            
            # Download the model
            show_download_progress(f"{provider} (spaCy dependency)", f"language model '{spacy_model}'")
            import subprocess
            import sys
            result = subprocess.run([
                sys.executable, '-m', 'spacy', 'download', spacy_model
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Failed to download spaCy model: {result.stderr}")
                return False
            
            print("Download complete")
            return True
    except ImportError:
        # spaCy not available, no consent needed
        return True


def with_download_progress(provider: str, model_name: str, download_func: Callable):
    """Wrapper that adds download progress and user consent."""
    def wrapper(*args, **kwargs):
        if not ask_user_consent(provider, model_name):
            raise RuntimeError(f"Download cancelled: {provider}/{model_name}")
        
        show_download_progress(provider, model_name)
        result = download_func(*args, **kwargs)
        print("Download complete")
        return result
    
    return wrapper


def check_model_cached(provider: str, model_name: str) -> bool:
    """Check if a model is already cached locally."""
    try:
        if provider == 'local_whisper':
            from faster_whisper.utils import download_model
            download_model(model_name, local_files_only=True)
            return True
        elif provider == 'pywhispercpp':
            from pywhispercpp.constants import MODELS_DIR
            model_path = os.path.join(MODELS_DIR, f'ggml-{model_name}.bin')
            return os.path.exists(model_path)
        elif provider == 'kittentts':
            from transformers import cached_file
            cached_file(model_name, "config.json", local_files_only=True)
            return True
        elif provider == 'kokoro':
            # Kokoro uses HuggingFace cache for the hexgrad/Kokoro-82M model
            cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub' / 'models--hexgrad--Kokoro-82M'
            return cache_dir.exists() and any(cache_dir.iterdir())
    except Exception:
        pass
    return False