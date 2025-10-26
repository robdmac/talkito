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

"""Claude integration initialization for Talkito - handles setting up Claude Desktop configuration and permissions"""

import io
import json
import logging
from pathlib import Path
import re
import socket
import subprocess
import shutil
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request

import talkito.logs

from .api import start_api_server
from .core import build_comms_config
from .logs import log_message, setup_logging
from .mcp import app, configure_mcp_server, find_available_port
from .state import get_status_summary, show_tap_to_talk_notification_once, sync_communication_state_from_config, get_shared_state
from .templates import ENV_EXAMPLE_TEMPLATE


TALKITO_PERMISSIONS = [
    "mcp__talkito__speak_text",
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
    "mcp__talkito__disable_asr",
    "mcp__talkito__list_tts_voices",
    "mcp__talkito__randomize_tts_voice",
    "mcp__talkito__cycle_tts_voice",
    "mcp__talkito__set_asr_mode",
    "mcp__talkito__set_tts_mode",
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
    for perm in TALKITO_PERMISSIONS:
        if perm not in settings["permissions"]["allow"]:
            settings["permissions"]["allow"].append(perm)
    
    # Save updated settings
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=2)
    
    return True


def update_claude_hooks(webhook_port=8080):
    """Update Claude hooks to use webhook server with the correct port"""
    settings_file = Path(".claude") / "settings.local.json"
    
    # Load existing settings
    settings = {}
    if settings_file.exists():
        try:
            with open(settings_file, 'r') as f:
                content = f.read().strip()
                if content:
                    settings = json.loads(content)
        except Exception as e:
            print(f"Warning: Could not load existing settings.json: {e}")
            return False
    
    # Define hook types needed for making sure we ask questions to the user
    hook_types = ["PreToolUse", "PostToolUse"]
    
    # Create hooks structure if it doesn't exist
    if "hooks" not in settings:
        settings["hooks"] = {}
    
    # Update each hook to use curl command with the correct port
    for hook_type in hook_types:
        curl_cmd = f'curl -s -X POST http://127.0.0.1:{webhook_port}/hook -H \'Content-Type: application/json\' -d \'{{\"hook_type\": \"{hook_type}\", \"timestamp\": \"\'$(date -u +%Y-%m-%dT%H:%M:%SZ)\'\"}}\' > /dev/null 2>&1 || true'
        
        settings["hooks"][hook_type] = [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": curl_cmd
                    }
                ]
            }
        ]
    
    # Save updated settings
    try:
        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        print(f"Error updating hooks: {e}")
        return False


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


def find_talkito_command():
    """Find the talkito command path"""
    # First try 'which' command
    try:
        result = subprocess.run(['which', 'talkito'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    
    # Fall back to Python's shutil.which
    path = shutil.which('talkito')
    if path:
        return path
    
    # Default fallback
    return "talkito"


def init_claude(address="http://127.0.0.1", port=8001):
    """Initialize Claude integration for talkito (uses streamable-http transport)"""

    success = True

    try:
        update_claude_settings()

        # Check if claude CLI is available
        claude_path = shutil.which('claude')
        if claude_path:
            mcp_url = f"{address}:{port}/mcp"

            # Remove old config if it exists
            subprocess.run(['claude', 'mcp', 'remove', 'talkito'],
                         capture_output=True, text=True)

            result = subprocess.run(
                ['claude', 'mcp', 'add', '--transport', 'http', 'talkito', mcp_url],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                print(f"Failed to add MCP server: {result.stderr}")
                success = False
            else:
                log_message("DEBUG", f"Configured Claude to use talkito MCP server at {mcp_url} (streamable-http)")
        else:
            print("Claude CLI not found")
            print("Install Claude CLI and run: claude mcp add --transport http talkito http://127.0.0.1:8000/mcp")
            success = False
    except Exception as e:
        print(f"Error adding MCP server: {e}")
        success = False

    return success


def init_codex(transport="streamable-http", address="http://127.0.0.1", port=8001):
    """Initialize Codex integration for talkito (uses streamable-http transport)"""

    # Check if codex CLI is available
    codex_path = shutil.which('codex')
    if not codex_path:
        print("Codex CLI not found")
        print("Install Codex CLI - see: https://developers.openai.com/codex/")
        return False

    try:
        mcp_url = f"{address}:{port}/mcp"

        codex_config_path = Path.home() / ".codex" / "config.toml"
        codex_config_path.parent.mkdir(exist_ok=True)

        # Read existing config
        config_content = ""
        if codex_config_path.exists():
            with open(codex_config_path, 'r') as f:
                config_content = f.read()

        # Remove old talkito section if exists
        talkito_section_pattern = r'\[mcp_servers\.talkito\].*?(?=\n\[|$)'
        config_content = re.sub(talkito_section_pattern, '', config_content, flags=re.DOTALL).strip()

        # Add new section for streamable-http
        new_section = f'\n\n[mcp_servers.talkito]\nurl = "{mcp_url}"\n'
        config_content += new_section

        # Write config back
        with open(codex_config_path, 'w') as f:
            f.write(config_content)

        log_message("DEBUG", f"Configured Codex to use talkito MCP server at {mcp_url} (streamable-http)")
        return True

    except Exception as e:
        print(f"Error adding MCP server: {e}")
        traceback.print_exc()
        return False


async def run_terminal_agent_extensions(args) -> int:
    """Run Claude with in-process MCP server and wrapper functionality"""

    log_message("DEBUG", "run_terminal_agent_extensions")

    # Determine if we should enable MCP server
    # Use getattr with default False to avoid AttributeError in test/utility code
    mcp_enabled = not getattr(args, "disable_mcp", False)
    port = None

    # Find available port only if MCP is enabled
    if mcp_enabled:
        port = args.port if args.port else find_available_port(8000)
        if not port:
            print("Warning: Could not find an available port for MCP server - continuing without MCP extensions", file=sys.stderr)
            mcp_enabled = False
            get_shared_state().set_mcp_server_running(False)
    else:
        # MCP explicitly disabled - ensure state reflects this
        get_shared_state().set_mcp_server_running(False)

    # Start MCP server in background thread (only if enabled)
    # Always create Event object to avoid AttributeError if code is refactored
    server_ready = threading.Event()

    def check_port_listening(host='127.0.0.1', check_port=None, timeout=0.1):
        """Check if a port is listening"""
        check_port = check_port or port  # Use the outer scope port if not specified
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(timeout)
                result = sock.connect_ex((host, check_port))
                return result == 0
        except Exception:
            return False
    
    def optimized_mcp_health_check(check_port, timeout=10):
        """
        Fast health check for MCP server using streamable-http endpoint
        Returns True when server is ready, False on timeout
        """
        start_time = time.time()
        last_check_time = 0
        check_interval = 0.05  # Start with 50ms checks
        max_check_interval = 0.2  # Cap at 200ms

        # First wait for port to be listening
        while time.time() - start_time < timeout:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(0.1)
                    if s.connect_ex(('127.0.0.1', check_port)) == 0:
                        break
            except Exception:
                pass
            time.sleep(0.05)
        else:
            return False  # Port never started listening

        # Now do progressive health checks on /mcp endpoint
        while time.time() - start_time < timeout:
            current_time = time.time()

            # Only check if enough time has passed (progressive backoff)
            if current_time - last_check_time >= check_interval:
                try:
                    # Try /mcp endpoint - responds immediately when ready
                    req = urllib.request.Request(
                        f'http://127.0.0.1:{check_port}/mcp',
                        headers={'Accept': 'application/json'}
                    )
                    with urllib.request.urlopen(req, timeout=0.3):
                        # Server is ready!
                        return True
                except urllib.error.HTTPError as e:
                    # HTTP errors indicate server is responding
                    if 200 <= e.code < 500:
                        return True
                except Exception:
                    # Connection errors - server not ready yet
                    pass

                last_check_time = current_time
                # Progressive backoff - increase interval slightly each time
                check_interval = min(check_interval * 1.1, max_check_interval)

            time.sleep(0.02)  # Small sleep to avoid busy waiting

        return False  # Timeout
    
    def run_mcp_server():
        """Run the MCP server in a thread"""
        try:
            # Configure MCP server through public interface
            
            # Build configuration
            config = {'cors_enabled': True}
            
            config['running_for_terminal_agent'] = True
            
            if args.log_file:
                config['log_file_path'] = args.log_file
                
            if not args.dont_auto_skip_tts:
                config['auto_skip_tts'] = True
            
            # Apply configuration
            configure_mcp_server(**config)
            
            # Apply filter to mcp tools
            apply_terminal_code_agent_tool_filter()
            
            # Start a thread to check when server is actually listening
            def monitor_server_startup():
                start_time = time.time()
                
                # First wait for port to be listening
                while time.time() - start_time < 10:  # 10 second timeout
                    if check_port_listening('127.0.0.1', port):
                        break
                    time.sleep(0.1)
                else:
                    # Timeout waiting for port
                    return
                
                # Then wait for FastMCP to be ready to accept connections
                # The port being open doesn't mean FastMCP is fully initialized
                # We'll do a simple check and then add a small delay
                while time.time() - start_time < 15:  # 15 second total timeout
                    try:
                        # Just try a simple GET request to see if server responds
                        req = urllib.request.Request(f'http://127.0.0.1:{port}/')
                        with urllib.request.urlopen(req, timeout=1):
                            # Server is responding
                            # Give FastMCP a bit more time to fully initialize its internals
                            # This is the key - even when responding, it needs more time
                            time.sleep(1.5)
                            server_ready.set()
                            return
                    except urllib.error.HTTPError as e:
                        # HTTP errors (4xx, 5xx) mean server is at least responding
                        if 400 <= e.code < 600:
                            # Server is up and responding, just not to this endpoint
                            time.sleep(1.5)  # Give it time to fully initialize
                            server_ready.set()
                            return
                    except Exception:
                        # Connection refused or timeout - server not ready yet
                        pass
                    
                    time.sleep(0.3)  # Wait before retrying
            
            monitor_thread = threading.Thread(target=monitor_server_startup, daemon=True)
            monitor_thread.start()
            
            # Temporarily redirect stderr to capture startup messages
            old_stderr = sys.stderr
            stderr_capture = io.StringIO()
            sys.stderr = stderr_capture
            # Use streamable-http for both Claude and Codex
            transport = "streamable-http"
            log_message("DEBUG", f"Starting MCP server on port {port} with transport={transport}")
            try:
                # Run the FastMCP server (this blocks)
                app.run(
                    transport=transport,
                    host="127.0.0.1",
                    port=port,
                    log_level="error"  # Changed from warning to error
                )
            except Exception as e:
                # Restore stderr first so we can print
                sys.stderr = old_stderr
                
                # Get captured stderr content
                # stderr_content = stderr_capture.getvalue()  # Captured but not used
                print(f"MCP server error: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                
                # Don't set server_ready on error - let it timeout
                return
            finally:
                # Restore stderr if not already restored
                if sys.stderr is not old_stderr:
                    sys.stderr = old_stderr
        except Exception as e:
            print(f"MCP server thread error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    try:
        # Determine webhook port - use args if provided, otherwise find available port
        if hasattr(args, 'webhook_port') and args.webhook_port:
            webhook_port = args.webhook_port
        else:
            # Find available port for API server (start from 8080)
            webhook_port = find_available_port(8080)
            if not webhook_port:
                print("Error: Could not find an available port for API server", file=sys.stderr)
                return 1

        # Set initial webhook_port for args (will be updated by background thread)
        args.webhook_port = webhook_port

        if args.command == 'claude':
            args = run_api_server(args)

        # Start the MCP server thread
        if mcp_enabled:
            log_message("INFO", "About to start MCP server thread")
            server_thread = threading.Thread(target=run_mcp_server, daemon=True)
            thread_start_time = time.time()
            server_thread.start()
            log_message("INFO", f"MCP server thread started [{time.time() - thread_start_time:.3f}s]")

            # Wait for server to be ready (max 15 seconds)
            if not server_ready.wait(15):
                print("Warning: MCP server startup timeout - server may not be fully initialized", file=sys.stderr)
                # Double-check if server is actually running despite timeout
                if not check_port_listening('127.0.0.1', port):
                    print("Error: MCP server failed to start on port", port, file=sys.stderr)
                    return 1
                else:
                    # Port is listening but server might not be fully ready
                    print("Note: MCP server port is listening but may still be initializing", file=sys.stderr)
                    # Use optimized health check instead of fixed 2-second sleep
                    health_check_start = time.time()
                    if optimized_mcp_health_check(port, timeout=3):
                        log_message("INFO", f"MCP server health check passed [{time.time() - health_check_start:.3f}s]")
                    else:
                        log_message("WARNING", f"MCP server health check timeout after {time.time() - health_check_start:.3f}s")
                        # Fall back to small sleep if health check fails
                        time.sleep(0.5)

                    # Mark MCP server as running (port is listening even if health check had issues)
                    get_shared_state().set_mcp_server_running(True)
            else:
                # Server signaled ready - give it a tiny bit more time to be safe
                time.sleep(0.2)
                # Mark MCP server as running in shared state
                get_shared_state().set_mcp_server_running(True)

            log_message("INFO", "restoring logging")
            # Restore logging after FastMCP has messed with it
            if args.log_file:
                # Force reset of logging configuration
                talkito.logs._is_configured = False

                # Clear all handlers that uvicorn/fastmcp added
                root_logger = logging.getLogger()
                root_logger.handlers.clear()

                # Re-setup logging
                setup_logging(args.log_file, mode='a')

        try:
            create_talkito_env()
        except Exception as e:
            print(f"Error creating .talkito.env: {e}")

        # Initialize coding terminal agent with streamable-http transport (only if MCP is enabled)
        if mcp_enabled:
            if args.command == 'claude':
                # Claude uses streamable-http transport
                if not init_claude(address="http://127.0.0.1", port=port):
                    print("Warning: Failed to configure Claude Code for streamable-http", file=sys.stderr)
            elif args.command == 'codex':
                # Codex uses streamable-http transport
                if not init_codex(address="http://127.0.0.1", port=port):
                    print("Warning: Failed to configure Codex CLI for streamable-http", file=sys.stderr)
        else:
            log_message("INFO", "MCP server disabled - skipping agent configuration")

        log_message("INFO", "starting print_configuration_status")
        # Show configuration status (now reflects actual providers after fallback)
        print_configuration_status(args)
        log_message("INFO", "print_configuration_status completed")

        return True

    except Exception as e:
        print(f"Error running TalkiTo Claude extensions: {e}")

        return False

def run_api_server(args):
    # Start the API server first (for webhooks and Claude hooks)
    step_start = time.time()
    log_message("INFO", f"API server port discovery completed [{time.time() - step_start:.3f}s]")

    # Start API server in background thread (non-critical for immediate startup)
    def start_api_server_background():
        step_start = time.time()
        log_message("INFO", "Starting API server in background...")
        api_server = start_api_server(port=args.webhook_port)
        log_message("INFO", f"API server startup completed [{time.time() - step_start:.3f}s]")

        # Get the actual port the server is running on (might be different if original was in use)
        actual_webhook_port = args.webhook_port
        if api_server:
            actual_port = api_server.get_port()
            if actual_port != args.webhook_port:
                actual_webhook_port = actual_port
                log_message("INFO",
                            f"API server using port {actual_port} instead of requested {args.webhook_port}")
        else:
            print("Warning: API server failed to start", file=sys.stderr)
            actual_webhook_port = None

        # Update Claude hooks to use the API server
        if actual_webhook_port:
            try:
                update_claude_hooks(actual_webhook_port)
                log_message("INFO", f"Claude hooks updated for port {actual_webhook_port}")
            except Exception as e:
                log_message("ERROR", f"Failed to update Claude hooks: {e}")

        # Update args with the webhook port so comms can use it
        args.webhook_port = actual_webhook_port

    # Start API server in background thread
    api_thread = threading.Thread(target=start_api_server_background, daemon=True, name="APIServerThread")
    api_thread.start()
    log_message("INFO", "API server thread started in background")
    return args

def apply_terminal_code_agent_tool_filter():
    """Apply monkey patch to filter tools for terminal coding agents that call tts/asr directly"""
    # Store the original method from the tool manager
    original_list_tools = app._tool_manager.list_tools
    
    # Define our filtered version
    async def filtered_list_tools():
        # Get the full list of tools
        all_tools = await original_list_tools()
        
        # Filter out tools that should be masked for Claude
        tools_to_mask = {
            'get_dictated_text', 
            'send_whatsapp',
            'send_slack',
            'get_messages',
            'start_notification_stream'
        }
        
        # The tools are Tool objects with a 'key' attribute
        filtered_tools = [tool for tool in all_tools if tool.key not in tools_to_mask]
        
        return filtered_tools
    
    # Replace the method on the tool manager
    app._tool_manager.list_tools = filtered_list_tools


def print_configuration_status(args):
    """Print the current TTS/ASR and communication configuration"""

    # Preview communication configuration for status banner
    comms_config = build_comms_config(args)
    sync_communication_state_from_config(comms_config)

    # Show one-time notification about tap-to-talk change if needed
    show_tap_to_talk_notification_once()
    
    # Don't pass configured providers to allow showing actual working providers after fallback
    status = get_status_summary(
        tts_override=True, 
        asr_override=(args.asr_mode != "off")
    )

    # Print with the same format but add the note about .talkito.env
    print(status)


if __name__ == "__main__":
    # For testing
    init_claude()
