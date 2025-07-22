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

"""
Claude integration initialization for Talkito
Handles setting up Claude Desktop configuration and permissions
"""

import json
from pathlib import Path
import subprocess
import shutil

from .templates import ENV_EXAMPLE_TEMPLATE, TALKITO_MD_CONTENT


TALKITO_PERMISSIONS = [
    # "mcp__talkito__speak_text",
    "mcp__talkito__skip_current_speech",
    "mcp__talkito__get_speech_status",
    "mcp__talkito__wait_for_speech_completion",
    "mcp__talkito__change_asr",
    "mcp__talkito__change_tts",
    "mcp__talkito__configure_asr",
    "mcp__talkito__configure_tts",
    "mcp__talkito__start_voice_input",
    "mcp__talkito__stop_voice_input",
    # "mcp__talkito__get_voice_input_status",
    # "mcp__talkito__get_dictated_text",
    "mcp__talkito__turn_on",
    "mcp__talkito__turn_off",
    "mcp__talkito__configure_communication",
    "mcp__talkito__send_whatsapp",
    "mcp__talkito__send_slack",
    "mcp__talkito__get_communication_status",
    "mcp__talkito__start_whatsapp_mode",
    "mcp__talkito__stop_whatsapp_mode",
    "mcp__talkito__get_whatsapp_mode_status",
    "mcp__talkito__start_slack_mode",
    "mcp__talkito__stop_slack_mode",
    "mcp__talkito__get_slack_mode_status",
    "mcp__talkito__get_messages",
    # "mcp__talkito__start_notification_stream",
    "mcp__talkito__get_talkito_status",
    "mcp__talkito__enable_tts",
    "mcp__talkito__disable_tts",
    "mcp__talkito__enable_asr",
    "mcp__talkito__disable_asr"
]


def update_claude_settings():
    """Update local Claude settings with talkito permissions"""
    settings_file = Path(".claude") / "settings.local.json"
    
    # Create .claude directory if it doesn't exist
    settings_file.parent.mkdir(exist_ok=True)
    
    # Load existing settings or create new structure
    settings = {}
    if settings_file.exists():
        try:
            with open(settings_file, 'r') as f:
                content = f.read().strip()
                if content:
                    settings = json.loads(content)
        except Exception as e:
            print(f"Warning: Could not load existing settings.json: {e}")
            settings = {}
    
    # Ensure permissions structure exists
    if "permissions" not in settings:
        settings["permissions"] = {"allow": [], "deny": []}
    if "allow" not in settings["permissions"]:
        settings["permissions"]["allow"] = []
    
    # Add permissions that aren't already present
    added = 0
    for perm in TALKITO_PERMISSIONS:
        if perm not in settings["permissions"]["allow"]:
            settings["permissions"]["allow"].append(perm)
            added += 1
    
    # Save updated settings
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=2)
    
    return True


def create_talkito_md():
    """Create TALKITO.md in current directory"""
    talkito_md_path = Path("TALKITO.md")
    
    exists = talkito_md_path.exists()

    if exists:
        print("TALKITO.md already exists")
        return True
    
    with open(talkito_md_path, 'w') as f:
        f.write(TALKITO_MD_CONTENT)
    
    # if existed:
    #     print("✓ Rewrote TALKITO.md")
    # else:

    print("Created TALKITO.md")

    return True


def create_talkito_env():
    """Create .talkito.env template in current directory"""
    talkito_env_path = Path(".talkito.env")
    
    # Skip if .talkito.env already exists
    if talkito_env_path.exists():
        return True
    
    with open(talkito_env_path, 'w') as f:
        f.write(ENV_EXAMPLE_TEMPLATE)
    
    print("Created .talkito.env (copy settings to .env as needed)")
    return True


def update_claude_md():
    """Update or create CLAUDE.md with talkito import"""
    claude_md_path = Path("CLAUDE.md")
    
    if claude_md_path.exists():
        # Check if talkito section already exists
        with open(claude_md_path, 'r') as f:
            content = f.read()
        
        if "@TALKITO.md" in content:
            print("CLAUDE.md already contains talkito import")
            return True  # Already exists, nothing to do
        else:
            # Add talkito section to existing CLAUDE.md
            with open(claude_md_path, 'a') as f:
                f.write("\n## Talkito MCP Voice Mode\n")
                f.write("- @TALKITO.md\n")
            print("Updated CLAUDE.md with talkito import")
    else:
        # Create new CLAUDE.md with talkito section
        content = """# CLAUDE.md

This file provides guidance to Claude Code when working in this project.

## Talkito MCP Voice Mode
- @TALKITO.md
"""
        with open(claude_md_path, 'w') as f:
            f.write(content)
        print("Created CLAUDE.md with talkito import")
    
    return True


def find_talkito_command():
    """Find the talkito command path"""
    # First try 'which' command
    try:
        result = subprocess.run(['which', 'talkito'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    
    # Fall back to Python's shutil.which
    import shutil
    path = shutil.which('talkito')
    if path:
        return path
    
    # Default fallback
    return "talkito"




def init_claude(transport="sse", address="http://127.0.0.1", port=8001):
    """Initialize Claude integration for talkito"""
    
    success = True
    
    # 1. Create TALKITO.md
    # try:
    #     create_talkito_md()
    # except Exception as e:
    #     print(f"✗ Error creating TALKITO.md: {e}")
    #     success = False
    
    # 2. Update Claude settings
    try:
        update_claude_settings()
    except Exception as e:
        print(f"✗ Error updating Claude settings: {e}")
        success = False
    
    # 3. Update CLAUDE.md
    # try:
    #     update_claude_md()
    # except Exception as e:
    #     print(f"✗ Error updating CLAUDE.md: {e}")
    #     success = False
    
    # 4. Create .talkito.env template
    try:
        create_talkito_env()
    except Exception as e:
        print(f"✗ Error creating .talkito.env: {e}")
        success = False
    
    # 5. Add talkito to Claude MCP servers
    try:
        # Check if claude CLI is available
        claude_path = shutil.which('claude')
        if claude_path:
            # Run the claude mcp add command
            mcp_url = None  # Initialize for scope
            if transport == "sse":
                mcp_url = address+str(port)+"/sse"
                result = subprocess.run(
                    ['claude', 'mcp', 'add', 'talkito', mcp_url, '--transport', transport,],
                    capture_output=True,
                    text=True
                )
            else:
                result = subprocess.run(
                    ['claude', 'mcp', 'add', 'talkito', 'talkito', '--', '--mcp-server'],
                    capture_output=True,
                    text=True
                )
            if result.returncode != 0:
                if "already exists" in result.stderr:
                    # Try to update the configuration by removing and re-adding
                    remove_result = subprocess.run(
                        ['claude', 'mcp', 'remove', 'talkito'],
                        capture_output=True,
                        text=True
                    )
                    if remove_result.returncode == 0:
                        # Now re-add with the new configuration
                        if transport == "sse":
                            result = subprocess.run(
                                ['claude', 'mcp', 'add', 'talkito', mcp_url, '--transport', transport,],
                                capture_output=True,
                                text=True
                            )
                            if result.returncode != 0:
                                print(f"✗ Failed to re-add MCP server: {result.stderr}")
                                success = False
                else:
                    print(f"✗ Failed to add MCP server: {result.stderr}")
                    success = False
        else:
            print("✗ Claude CLI not found")
            print("Install Claude CLI and run: claude mcp add talkito talkito -- --mcp-server")
            success = False
    except Exception as e:
        print(f"✗ Error adding MCP server: {e}")
        success = False
    
    return success


if __name__ == "__main__":
    # For testing
    init_claude()