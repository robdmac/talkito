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

import json
from pathlib import Path
import subprocess
import shutil
import sys
import threading
import time

from .templates import ENV_EXAMPLE_TEMPLATE, TALKITO_MD_CONTENT
from .state import get_status_summary, show_tap_to_talk_notification_once
from .logs import log_message


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
    added = 0
    for perm in TALKITO_PERMISSIONS:
        if perm not in settings["permissions"]["allow"]:
            settings["permissions"]["allow"].append(perm)
            added += 1
    
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
    except Exception:
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
                    ['claude', 'mcp', 'add', 'talkito', mcp_url, '--transport', transport],
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


async def run_claude_extensions(args) -> int:
    """Run Claude with in-process MCP server and wrapper functionality"""
    from .mcp import app, find_available_port

    log_message("DEBUG", "run_claude_hybrid")
    
    # Find available port
    port = args.port if args.port else find_available_port(8000)
    if not port:
        print("Error: Could not find an available port", file=sys.stderr)
        return 1

    # Start MCP server in background thread
    server_ready = threading.Event()
    server_thread = None
    
    def check_port_listening(host='127.0.0.1', check_port=None, timeout=0.1):
        """Check if a port is listening"""
        import socket
        check_port = check_port or port  # Use the outer scope port if not specified
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, check_port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def optimized_mcp_health_check(check_port, timeout=10):
        """
        Fast health check for MCP server using SSE endpoint
        Returns True when server is ready, False on timeout
        """
        import socket
        import urllib.request
        import urllib.error
        
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
        
        # Now do progressive health checks on SSE endpoint
        while time.time() - start_time < timeout:
            current_time = time.time()
            
            # Only check if enough time has passed (progressive backoff)
            if current_time - last_check_time >= check_interval:
                try:
                    # Try SSE endpoint - responds immediately when ready
                    req = urllib.request.Request(
                        f'http://127.0.0.1:{check_port}/sse',
                        headers={'Accept': 'text/event-stream'}
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
            from . import mcp
            
            # Build configuration
            config = {'cors_enabled': True}
            
            if args.command == 'claude':
                config['running_for_claude'] = True
            
            if args.log_file:
                config['log_file_path'] = args.log_file
                
            if not args.dont_auto_skip_tts:
                config['auto_skip_tts'] = True
            
            # Apply configuration
            mcp.configure_mcp_server(**config)
            
            # Apply monkey patch for Claude to filter tools
            apply_claude_tool_filter()
            
            # Start a thread to check when server is actually listening
            def monitor_server_startup():
                import urllib.request
                import urllib.error
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
            import io
            old_stderr = sys.stderr
            stderr_capture = io.StringIO()
            sys.stderr = stderr_capture
            
            try:
                # Run the FastMCP server (this blocks)
                app.run(
                    transport="sse",
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
                import traceback
                traceback.print_exc(file=sys.stderr)
                
                # Don't set server_ready on error - let it timeout
                return
            finally:
                # Restore stderr if not already restored
                if sys.stderr is not old_stderr:
                    sys.stderr = old_stderr
        except Exception as e:
            print(f"MCP server thread error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
    
    try:
        # Start the API server first (for webhooks and Claude hooks)
        step_start = time.time()
        from .api import start_api_server
        
        # Determine webhook port - use args if provided, otherwise find available port
        if hasattr(args, 'webhook_port') and args.webhook_port:
            webhook_port = args.webhook_port
        else:
            # Find available port for API server (start from 8080)
            webhook_port = find_available_port(8080)
            if not webhook_port:
                print("Error: Could not find an available port for API server", file=sys.stderr)
                return 1
        
        log_message("INFO", f"[CLAUDE_HYBRID] API server port discovery completed [{time.time() - step_start:.3f}s]")
        
        # Start API server in background thread (non-critical for immediate startup)
        def start_api_server_background():
            step_start = time.time()
            log_message("INFO", "[CLAUDE_HYBRID] Starting API server in background...")
            api_server = start_api_server(port=webhook_port)
            log_message("INFO", f"[CLAUDE_HYBRID] API server startup completed [{time.time() - step_start:.3f}s]")
            
            # Get the actual port the server is running on (might be different if original was in use)
            actual_webhook_port = webhook_port
            if api_server:
                actual_port = api_server.get_port()
                if actual_port != webhook_port:
                    actual_webhook_port = actual_port
                    log_message("INFO", f"[CLAUDE_HYBRID] API server using port {actual_port} instead of requested {webhook_port}")
            else:
                print("Warning: API server failed to start", file=sys.stderr)
                actual_webhook_port = None

            # Update Claude hooks to use the API server
            if actual_webhook_port:
                try:
                    update_claude_hooks(actual_webhook_port)
                    log_message("INFO", f"[CLAUDE_HYBRID] Claude hooks updated for port {actual_webhook_port}")
                except Exception as e:
                    log_message("ERROR", f"[CLAUDE_HYBRID] Failed to update Claude hooks: {e}")
            
            # Update args with the webhook port so comms can use it
            args.webhook_port = actual_webhook_port
        
        # Start API server in background thread
        api_thread = threading.Thread(target=start_api_server_background, daemon=True, name="APIServerThread")
        api_thread.start()
        log_message("INFO", "[CLAUDE_HYBRID] API server thread started in background")
        
        # Set initial webhook_port for args (will be updated by background thread)
        args.webhook_port = webhook_port
        
        # Start the MCP server thread
        log_message("INFO", "[CLAUDE_HYBRID] About to start MCP server thread")
        server_thread = threading.Thread(target=run_mcp_server, daemon=True)
        thread_start_time = time.time()
        server_thread.start()
        log_message("INFO", f"[CLAUDE_HYBRID] MCP server thread started [{time.time() - thread_start_time:.3f}s]")
        
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
                    log_message("INFO", f"[CLAUDE_HYBRID] MCP server health check passed [{time.time() - health_check_start:.3f}s]")
                else:
                    log_message("WARNING", f"[CLAUDE_HYBRID] MCP server health check timeout after {time.time() - health_check_start:.3f}s")
                    # Fall back to small sleep if health check fails
                    time.sleep(0.5)
        else:
            # Server signaled ready - give it a tiny bit more time to be safe
            time.sleep(0.2)
        log_message("INFO", "[CLAUDE_HYBRID] restoring logging")
        # Restore logging after FastMCP has messed with it
        if args.log_file:
            import logging
            from .logs import setup_logging
            import talkito.logs
            
            # Force reset of logging configuration
            talkito.logs._is_configured = False
            
            # Clear all handlers that uvicorn/fastmcp added
            root_logger = logging.getLogger()
            root_logger.handlers.clear()
            
            # Re-setup logging
            setup_logging(args.log_file, mode='a')

        # Initialize Claude with SSE configuration
        if not init_claude(transport="sse", address="http://127.0.0.1:", port=port):
            print("Warning: Failed to configure Claude for SSE", file=sys.stderr)

        log_message("INFO", "[CLAUDE_HYBRID] starting print_configuration_status")
        # Show configuration status (now reflects actual providers after fallback)
        print_configuration_status(args)
        log_message("INFO", "[CLAUDE_HYBRID] print_configuration_status completed")

        return True
        
    except Exception as e:
        print(f"Error running TalkiTo Claude extensions: {e}")

        return False


def apply_claude_tool_filter():
    """Apply monkey patch to filter tools for Claude"""
    from .mcp import app
    
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

    # Force state initialization by importing and accessing it
    # from .state import get_shared_state
    # shared_state = get_shared_state()  # This ensures the state singleton is initialized and loaded
    
    # Build comms config to check what's configured
    # comms_config = build_comms_config(args)
    #
    # # Update shared state with configured providers from args/env
    # if comms_config:
    #     # Check if providers are configured
    #     has_whatsapp = bool(comms_config.twilio_whatsapp_number and comms_config.whatsapp_recipients)
    #     has_slack = bool(comms_config.slack_bot_token and comms_config.slack_app_token and comms_config.slack_channel)
    #
    #     # Update shared state
    #     shared_state.communication.whatsapp_enabled = has_whatsapp
    #     shared_state.communication.slack_enabled = has_slack
    #
    #     # Also update recipients/channels
    #     if has_whatsapp and comms_config.whatsapp_recipients:
    #         shared_state.communication.whatsapp_to_number = comms_config.whatsapp_recipients[0]
    #     if has_slack and comms_config.slack_channel:
    #         shared_state.communication.slack_channel = comms_config.slack_channel
    
    # Show one-time notification about tap-to-talk change if needed
    show_tap_to_talk_notification_once()
    
    # Don't pass configured providers to allow showing actual working providers after fallback
    status = get_status_summary(
        tts_override=True, 
        asr_override=(args.asr_mode != "off")
    )

    # Print with the same format but add the note about .talkito.env
    print(f"╭ {status}")


if __name__ == "__main__":
    # For testing
    init_claude()