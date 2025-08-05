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
import os
import sys
import asyncio
import time
import threading
import signal
from typing import Optional

from .templates import ENV_EXAMPLE_TEMPLATE, TALKITO_MD_CONTENT
from . import tts
from . import asr
from . import comms
from .core import run_with_talkito
from .state import get_status_summary, get_shared_state


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
    "mcp__talkito__cycle_tts_voice"
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
    
    # Define hook types
    hook_types = [
        "SessionStart", "PreToolUse", "PostToolUse", "Notification",
        "UserPromptSubmit", "Stop", "SubagentStop", "PreCompact"
    ]
    
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


async def run_claude_wrapper(args) -> int:
    """Run Claude with wrapper mode"""
    cmd = [args.command] + args.arguments

    # Build kwargs for run_with_talkito
    kwargs = {
        'verbosity': args.verbose,
        'asr_mode': args.asr_mode,
        'record_file': args.record,
        'auto_skip_tts': not args.dont_auto_skip_tts,  # Auto-skip is on by default, disabled with --dont-auto-skip-tts
    }

    # Add other configurations...
    if args.log_file:
        kwargs['log_file'] = args.log_file

    if args.profile:
        kwargs['profile'] = args.profile
    else:
        kwargs['profile'] = 'claude'

    # Add TTS config
    tts_config = build_tts_config(args)
    if tts_config:
        kwargs['tts_config'] = tts_config

    # Add ASR config
    if asr:
        asr_config = build_asr_config(args)
        if asr_config:
            kwargs['asr_config'] = asr_config

    # Add communications config
    comms_config = build_comms_config(args)
    if comms_config:
        kwargs['comms_config'] = comms_config

    # Handle TTS disable
    if args.disable_tts:
        tts.disable_tts = True

    # Use the high-level API from core
    return await run_with_talkito(cmd, **kwargs)


async def run_claude_hybrid(args) -> int:
    """Run Claude with in-process MCP server and wrapper functionality"""
    import threading
    import time
    import signal
    from .mcp import app, find_available_port
    from .tts import stop_tts_immediately
    
    # Find available port
    port = args.port if args.port else find_available_port(8000)
    if not port:
        print("Error: Could not find an available port", file=sys.stderr)
        return 1
    
    # API port will be port + 1
    api_port = port + 1
    
    # Set environment variables for provider preferences
    if args.tts_provider:
        os.environ['TALKITO_PREFERRED_TTS_PROVIDER'] = args.tts_provider
    if args.asr_provider:
        os.environ['TALKITO_PREFERRED_ASR_PROVIDER'] = args.asr_provider
    
    # Start MCP server in background thread
    server_ready = threading.Event()
    server_thread = None
    
    # Set up signal handler for hybrid mode
    def hybrid_signal_handler(signum, frame):
        """Handle signals in hybrid mode - stop TTS immediately"""
        if signum == signal.SIGINT:
            try:
                stop_tts_immediately()
            except Exception:
                pass
            # Exit with conventional SIGINT exit code to avoid race conditions
            sys.exit(128 + signal.SIGINT)
    
    # Install our signal handler
    original_sigint_handler = signal.signal(signal.SIGINT, hybrid_signal_handler)
    
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
        except:
            return False
    
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
                import time
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
                        with urllib.request.urlopen(req, timeout=1) as response:
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
                stderr_content = stderr_capture.getvalue()
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
        
        # Start API server
        api_server = start_api_server(port=webhook_port)

        # Update Claude hooks to use the API server
        update_claude_hooks(webhook_port)
        
        # Update args with the webhook port so comms can use it
        args.webhook_port = webhook_port
        
        # Start the MCP server thread
        server_thread = threading.Thread(target=run_mcp_server, daemon=True)
        server_thread.start()
        
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
                # Give it a bit more time
                time.sleep(2)
        else:
            # Server signaled ready - give it a tiny bit more time to be safe
            time.sleep(0.2)
        
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
        
        # Show configuration status
        print_configuration_status(args)
        
        # Re-install our signal handler right before running Claude
        # This ensures it's the last one installed and will be called first
        signal.signal(signal.SIGINT, hybrid_signal_handler)
        
        # Now run Claude using the wrapper approach
        return await run_claude_wrapper(args)
        
    except Exception as e:
        print(f"Hybrid mode error: {e}")
        # Fall back to regular wrapper
        print("Falling back to standard wrapper mode...")
        return await run_claude_wrapper(args)
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_sigint_handler)


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


# Helper functions needed by the Claude runners
def build_tts_config(args) -> dict:
    """Build TTS configuration from command line arguments"""
    config = {}
    
    if args.tts_provider:
        config['provider'] = args.tts_provider
    else:
        # If no provider specified, use the best available one
        best_provider = tts.select_best_tts_provider()
        if best_provider != 'system':
            config['provider'] = best_provider
    if args.tts_voice:
        config['voice'] = args.tts_voice
    if args.tts_region:
        config['region'] = args.tts_region
    # Only include language if it's not the default or if a provider is specified
    if args.tts_language and (args.tts_language != 'en-US' or args.tts_provider):
        config['language'] = args.tts_language
    if args.tts_rate is not None:
        config['rate'] = args.tts_rate
    if args.tts_pitch is not None:
        config['pitch'] = args.tts_pitch
    
    # Only return config if we have actual values
    return config if config else None


def build_asr_config(args) -> dict:
    """Build ASR configuration from command line arguments"""
    config = {}
    
    if args.asr_provider:
        config['provider'] = args.asr_provider
    else:
        # If no provider specified, use the best available one
        if asr:
            try:
                best_provider = asr.select_best_asr_provider()
                if best_provider != 'google':  # 'google' is the free fallback
                    config['provider'] = best_provider
            except:
                pass
    if args.asr_language:
        config['language'] = args.asr_language
    if args.asr_model:
        config['model'] = args.asr_model
    
    return config if config else None


def build_comms_config(args) -> Optional[comms.CommsConfig]:
    """Build communication configuration from command line arguments"""
    config = comms.create_config_from_env()
    
    # Override with command line arguments first
    if args.sms_recipients:
        config.sms_recipients = [r.strip() for r in args.sms_recipients.split(',')]
    if args.whatsapp_recipients:
        config.whatsapp_recipients = [r.strip() for r in args.whatsapp_recipients.split(',')]
    if args.slack_channel:
        config.slack_channel = args.slack_channel
    if args.webhook_port:
        config.webhook_port = args.webhook_port
    
    # Check if any communication is configured (after applying overrides)
    has_sms = config.twilio_account_sid and config.sms_recipients
    has_whatsapp = config.twilio_whatsapp_number and config.whatsapp_recipients
    has_slack = config.slack_bot_token and config.slack_app_token and config.slack_channel
    
    if not any([has_sms, has_whatsapp, has_slack]):
        # No communication configured
        return None
    
    # Auto-detect based on configuration
    config.sms_enabled = has_sms
    config.whatsapp_enabled = has_whatsapp
    config.slack_enabled = has_slack
    
    return config


def print_configuration_status(args):
    """Print the current TTS/ASR and communication configuration"""

    # Force state initialization by importing and accessing it
    from .state import get_shared_state
    shared_state = get_shared_state()  # This ensures the state singleton is initialized and loaded
    
    # Build comms config to check what's configured
    comms_config = build_comms_config(args)
    
    # Update shared state with configured providers from args/env
    if comms_config:
        # Check if providers are configured
        has_whatsapp = bool(comms_config.twilio_whatsapp_number and comms_config.whatsapp_recipients)
        has_slack = bool(comms_config.slack_bot_token and comms_config.slack_app_token and comms_config.slack_channel)
        
        # Update shared state
        shared_state.communication.whatsapp_enabled = has_whatsapp
        shared_state.communication.slack_enabled = has_slack
        
        # Also update recipients/channels
        if has_whatsapp and comms_config.whatsapp_recipients:
            shared_state.communication.whatsapp_to_number = comms_config.whatsapp_recipients[0]
        if has_slack and comms_config.slack_channel:
            shared_state.communication.slack_channel = comms_config.slack_channel
    
    # Get the status summary using the shared function
    # Pass the configured providers from args if available
    configured_tts_provider = args.tts_provider if hasattr(args, 'tts_provider') and args.tts_provider else None
    configured_asr_provider = args.asr_provider if hasattr(args, 'asr_provider') and args.asr_provider else None
    
    status = get_status_summary(
        tts_override=True, 
        asr_override=(args.asr_mode != "off"),
        configured_tts_provider=configured_tts_provider,
        configured_asr_provider=configured_asr_provider
    )

    # Print with the same format but add the note about .talkito.env
    print(f"╭ {status}")


if __name__ == "__main__":
    # For testing
    init_claude()