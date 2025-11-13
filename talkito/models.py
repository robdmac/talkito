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
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

# Configure HuggingFace Hub timeouts via environment variables BEFORE any imports
# These are used by huggingface_hub when it loads its constants module
_hf_configured = False

def _configure_hf_timeouts():
    """Configure HuggingFace Hub timeouts based on whether models are likely cached."""
    global _hf_configured
    if _hf_configured:
        return

    try:
        # Import only constants to get the cache directory (respects HF_HUB_CACHE, HUGGINGFACE_HUB_CACHE, HF_HOME env vars)
        from huggingface_hub import constants

        cache_dir = Path(constants.HF_HUB_CACHE)

        if cache_dir.exists():
            cached_models = list(cache_dir.glob("models--*"))
            has_cache = len(cached_models) > 0
        else:
            has_cache = False

        # If we have cached models, use shorter timeout for HEAD requests (checking for updates)
        # If no cache, use longer timeout to allow first download
        etag_timeout = '5' if has_cache else '10'

        # Set environment variables (these are read by huggingface_hub.constants on import)
        os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', etag_timeout)  # HEAD request timeout
        os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '30')      # Actual download timeout

        _hf_configured = True
    except ImportError:
        # huggingface_hub not installed - skip configuration
        pass

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

def _hf_cached(repo_id: str, filename: Optional[str] = None,
               revision: Optional[str] = None,
               cache_dir: Optional[str] = None) -> bool:
    """Return True if a repo (or specific file) is already present in the local HF cache."""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        from huggingface_hub.utils import LocalEntryNotFoundError

        if filename:
            hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, local_files_only=True, cache_dir=cache_dir)
        else:
            snapshot_download(repo_id=repo_id, revision=revision, local_files_only=True, cache_dir=cache_dir)
        return True
    except LocalEntryNotFoundError:
        return False

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
            repo = model_name if '/' in model_name else f"KittenML/{model_name}"
            return _hf_cached(repo_id=repo, filename="config.json")
        elif provider == 'kokoro':
            repo = model_name if '/' in model_name else "hexgrad/Kokoro-82M"
            return _hf_cached(repo_id=repo)
        
    except Exception:
        pass
    return False