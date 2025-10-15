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

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError
from typing import Callable, Optional


def patch_hf_hub_with_timeout(connect_timeout: float = 5.0, read_timeout: float = 30.0):
    """
    Monkey patch huggingface_hub to use timeouts for network requests.

    This prevents hf_hub_download from hanging indefinitely when there's
    a network connection but no internet access.

    Args:
        connect_timeout: Timeout for establishing connection (seconds)
        read_timeout: Timeout for reading data (seconds)
    """
    try:
        import requests
        from huggingface_hub import file_download

        # Get the original get_hf_file_metadata and hf_hub_download functions
        original_get_session = getattr(file_download, 'get_session', None)

        if original_get_session:
            def patched_get_session(*args, **kwargs):
                """Patched version that sets default timeouts on requests."""
                session = original_get_session(*args, **kwargs)

                # Monkey patch the request method to always include timeout
                original_request = session.request

                def request_with_timeout(*req_args, **req_kwargs):
                    # Only add timeout if not already specified
                    if 'timeout' not in req_kwargs:
                        req_kwargs['timeout'] = (connect_timeout, read_timeout)
                    return original_request(*req_args, **req_kwargs)

                session.request = request_with_timeout
                return session

            file_download.get_session = patched_get_session

        # Also patch the constants module if available
        try:
            from huggingface_hub import constants
            if hasattr(constants, 'DEFAULT_REQUEST_TIMEOUT'):
                constants.DEFAULT_REQUEST_TIMEOUT = (connect_timeout, read_timeout)
        except (ImportError, AttributeError):
            pass

    except ImportError:
        # If requests or huggingface_hub internals changed, silently skip patching
        pass


# Apply the patch when module is imported
patch_hf_hub_with_timeout()

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

def _hf_cached(repo_id: str, filename: Optional[str] = None,
               revision: Optional[str] = None,
               cache_dir: Optional[str] = None) -> bool:
    """Return True if a repo (or specific file) is already present in the local HF cache."""
    try:
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