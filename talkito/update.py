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

"""Self-update functionality for talkito"""

import os
import sys
import subprocess
import tempfile
import shutil
import json
import time
import threading
from urllib.request import urlopen
from urllib.error import URLError
from pathlib import Path
from . import __version__
from .logs import get_logger

logger = get_logger(__name__)


class TalkitoUpdater:
    """Handles self-update functionality for talkito"""
    
    GITHUB_REPO_URL = "https://github.com/robdmac/talkito"
    UPDATE_CHECK_URL = "https://talkito-app-updates.robbomacrae.workers.dev/check-update"
    UPDATE_CHECK_INTERVAL = 3600*24  # Check every day
    
    def __init__(self):
        self.current_version = __version__
        self.install_dir = Path(__file__).parent.parent
        self.state_file = Path.home() / '.talkito_update_state.json'
        self.staging_dir = Path.home() / '.talkito_update_staging'
        self._background_thread = None
        self._stop_event = threading.Event()
        
    def check_for_updates(self):
        """Check if a newer version is available"""
        try:
            from urllib.parse import urlencode
            params = urlencode({
                'current': self.current_version,
                'method': self.get_update_method()
            })
            url = f"{self.UPDATE_CHECK_URL}?{params}"
            
            with urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode())
                latest_version = data['tag_name'].lstrip('v')
                return latest_version, self._compare_versions(latest_version, self.current_version) > 0
        except (URLError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to check for updates: {e}")
            return None, False
    
    def _compare_versions(self, v1, v2):
        """Compare two version strings (e.g., '1.2.3' vs '1.2.4')"""
        try:
            parts1 = [int(x) for x in v1.split('.')]
            parts2 = [int(x) for x in v2.split('.')]
            
            # Pad with zeros if needed
            max_len = max(len(parts1), len(parts2))
            parts1.extend([0] * (max_len - len(parts1)))
            parts2.extend([0] * (max_len - len(parts2)))
            
            for p1, p2 in zip(parts1, parts2):
                if p1 > p2:
                    return 1
                elif p1 < p2:
                    return -1
            return 0
        except Exception:
            # Fallback to string comparison
            return 1 if v1 > v2 else (-1 if v1 < v2 else 0)
    
    def get_update_method(self):
        """Determine how talkito was installed"""
        # Check if we're in a git repository
        git_dir = self.install_dir / '.git'
        if git_dir.exists():
            return 'git'
        
        # Check if installed via pip (in site-packages)
        if 'site-packages' in str(self.install_dir):
            return 'pip'
        
        # Check if in development mode (editable install)
        if (self.install_dir / 'setup.py').exists():
            return 'dev'
        
        return 'unknown'
    
    def update_via_git(self):
        """Update using git pull"""
        try:
            # Save current directory
            original_dir = os.getcwd()
            os.chdir(self.install_dir)
            
            # Stash any local changes
            subprocess.run(['git', 'stash'], capture_output=True)
            
            # Pull latest changes
            result = subprocess.run(['git', 'pull', 'origin', 'main'], 
                                    capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Git pull failed: {result.stderr}")
                # Try to pop stash
                subprocess.run(['git', 'stash', 'pop'], capture_output=True)
                return False
            
            # Pop stash if we had changes
            subprocess.run(['git', 'stash', 'pop'], capture_output=True)
            
            os.chdir(original_dir)
            return True
            
        except Exception as e:
            logger.error(f"Git update failed: {e}")
            return False
    
    def update_via_pip(self):
        """Update using pip"""
        try:
            # Use pip to upgrade
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', 
                                     '--upgrade', 'talkito'], 
                                    capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Pip upgrade failed: {result.stderr}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Pip update failed: {e}")
            return False
    
    def download_and_replace(self):
        """Download latest release and replace current installation"""
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Clone the repository
                result = subprocess.run(['git', 'clone', '--depth', '1', 
                                         self.GITHUB_REPO_URL + '.git', 
                                         str(temp_path / 'talkito')],
                                        capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Failed to clone repository: {result.stderr}")
                    return False
                
                # Copy new files over old ones
                source_dir = temp_path / 'talkito' / 'talkito'
                if source_dir.exists():
                    for item in source_dir.glob('*.py'):
                        dest = self.install_dir / item.name
                        shutil.copy2(item, dest)
                    
                    return True
                else:
                    logger.error("Downloaded repository structure unexpected")
                    return False
                    
        except Exception as e:
            logger.error(f"Download and replace failed: {e}")
            return False
    
    def perform_update(self):
        """Perform the update based on installation method"""
        method = self.get_update_method()
        logger.info(f"Update method: {method}")
        
        if method == 'git':
            return self.update_via_git()
        elif method == 'pip':
            return self.update_via_pip()
        elif method == 'dev':
            print("Development installation detected. Please update manually using git pull.")
            return False
        else:
            # Try direct download as fallback
            return self.download_and_replace()
    
    def update(self, force=False):
        """Main update function"""
        print(f"Current version: {self.current_version}")
        
        if not force:
            latest_version, update_available = self.check_for_updates()
            
            if latest_version is None:
                print("Unable to check for updates. Check your internet connection.")
                return False
            
            if not update_available:
                print(f"Already up to date (latest: {latest_version})")
                return True
            
            print(f"New version available: {latest_version}")
        else:
            print("Forcing update...")
        
        # Perform update
        success = self.perform_update()
        
        if success:
            print("\nUpdate completed successfully!")
            print("Please restart talkito to use the new version.")
            return True
        else:
            print("\nUpdate failed. Please check the logs or update manually.")
            return False
    
    def _load_state(self):
        """Load update state from file"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.debug(f"Failed to load update state: {e}")
        return {}
    
    def _save_state(self, state):
        """Save update state to file"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save update state: {e}")
    
    def should_check_for_updates(self):
        """Check if enough time has passed since last update check"""
        if os.environ.get('TALKITO_AUTO_UPDATE', 'true').lower() == 'false':
            return False
            
        state = self._load_state()
        last_check = state.get('last_check', 0)
        now = time.time()
        
        return (now - last_check) > self.UPDATE_CHECK_INTERVAL
    
    def stage_update(self, version):
        """Download and stage an update for next restart"""
        try:
            logger.info(f"Staging update to version {version}")
            
            # Create staging directory
            self.staging_dir.mkdir(exist_ok=True)
            
            # Clone to staging directory
            staging_repo = self.staging_dir / 'talkito'
            if staging_repo.exists():
                shutil.rmtree(staging_repo)
                
            result = subprocess.run(['git', 'clone', '--depth', '1', 
                                   '--branch', f'v{version}',
                                   self.GITHUB_REPO_URL + '.git', 
                                   str(staging_repo)],
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to clone repository: {result.stderr}")
                return False
            
            # Save staged version info
            state = self._load_state()
            state['staged_version'] = version
            state['staged_at'] = time.time()
            self._save_state(state)
            
            logger.info(f"Successfully staged version {version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stage update: {e}")
            return False
    
    def apply_staged_update(self):
        """Apply a staged update if available"""
        state = self._load_state()
        staged_version = state.get('staged_version')
        
        if not staged_version:
            return False
            
        staging_repo = self.staging_dir / 'talkito'
        if not staging_repo.exists():
            return False
            
        try:
            logger.info(f"Applying staged update to version {staged_version}")
            
            # Copy files from staging to install directory
            source_dir = staging_repo / 'talkito'
            if source_dir.exists():
                for item in source_dir.glob('*.py'):
                    dest = self.install_dir / item.name
                    shutil.copy2(item, dest)
                
                # Clean up staging
                shutil.rmtree(self.staging_dir)
                
                # Update state
                state = self._load_state()
                state.pop('staged_version', None)
                state.pop('staged_at', None)
                state['last_update'] = time.time()
                state['updated_to'] = staged_version
                self._save_state(state)
                
                logger.info(f"Successfully applied update to version {staged_version}")
                return True
            else:
                logger.error("Staged repository structure unexpected")
                return False
                
        except Exception as e:
            logger.error(f"Failed to apply staged update: {e}")
            return False
    
    def _background_check_loop(self):
        """Background thread that periodically checks for updates"""
        while not self._stop_event.is_set():
            try:
                if self.should_check_for_updates():
                    latest_version, update_available = self.check_for_updates()
                    
                    if update_available:
                        logger.info(f"Update available: {latest_version}")
                        # Stage the update in background
                        if self.stage_update(latest_version):
                            logger.info(f"Update {latest_version} staged for next restart")
                    
                    # Update last check time
                    state = self._load_state()
                    state['last_check'] = time.time()
                    self._save_state(state)
                    
            except Exception as e:
                logger.error(f"Error in background update check: {e}")
            
            # Sleep for a while, but check stop event regularly
            for _ in range(60):  # Check every minute
                if self._stop_event.wait(60):
                    break
    
    def start_background_updates(self):
        """Start background update checking"""
        if os.environ.get('TALKITO_AUTO_UPDATE', 'true').lower() == 'false':
            logger.info("Auto-updates disabled by environment variable")
            return
            
        if self._background_thread is None or not self._background_thread.is_alive():
            self._background_thread = threading.Thread(
                target=self._background_check_loop,
                daemon=True,
                name="TalkitoUpdateChecker"
            )
            self._background_thread.start()
            logger.info("Started background update checker")
    
    def stop_background_updates(self):
        """Stop background update checking"""
        if self._background_thread and self._background_thread.is_alive():
            self._stop_event.set()
            self._background_thread.join(timeout=5)
            logger.info("Stopped background update checker")


def check_and_apply_staged_update():
    """Check for and apply any staged updates on startup"""
    try:
        updater = TalkitoUpdater()
        if updater.apply_staged_update():
            print("Applied staged update. Talkito has been updated to the latest version.")
            return True
    except Exception as e:
        logger.error(f"Failed to apply staged update: {e}")
    return False


def start_background_update_checker():
    """Start the background update checker (called on startup)"""
    try:
        updater = TalkitoUpdater()
        updater.start_background_updates()
    except Exception as e:
        logger.error(f"Failed to start background update checker: {e}")


def main():
    """CLI entry point for update command"""
    import argparse
    parser = argparse.ArgumentParser(description='Update talkito to the latest version')
    parser.add_argument('--force', '-f', action='store_true', 
                        help='Force update even if already up to date')
    args = parser.parse_args()
    
    updater = TalkitoUpdater()
    success = updater.update(force=args.force)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()