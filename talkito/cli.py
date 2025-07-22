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

"""Talkito CLI - Command-line interface for the talkito package"""

import os
import argparse
import asyncio
import signal
import sys
from typing import List, Optional, Union, Tuple

# Suppress websockets deprecation warnings early
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="websockets")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="uvicorn")

# Try to load .env files if available
try:
    from dotenv import load_dotenv
    # Load .env first (takes precedence)
    load_dotenv()
    # Also load .talkito.env (won't override existing vars from .env)
    load_dotenv('.talkito.env')
except ImportError:
    # python-dotenv not installed, continue without it
    pass

# Import from our package
from . import __version__
from . import tts
from . import asr
from . import comms
from .core import TalkitoCore
from .state import get_status_summary

# Check Python version
if sys.version_info < (3, 8):
    print("Error: Python 3.8 or higher required", file=sys.stderr)
    print("Your version:", sys.version, file=sys.stderr)
    sys.exit(1)

# Global references for signal handlers
core_instance: Optional[TalkitoCore] = None


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    # Import cleanup function from core
    from .core import cleanup_terminal
    
    # First ensure we clean up the terminal - this is critical
    cleanup_terminal()
    
    # Then stop ASR and TTS
    if asr:
        try:
            asr.stop_dictation()
        except:
            pass
    
    # For interrupt signals, shutdown immediately without waiting
    if signum in (signal.SIGINT, signal.SIGTERM):
        try:
            tts.shutdown_tts()  # Shutdown immediately
        except:
            pass
    else:
        # For other signals, wait for TTS to finish
        try:
            tts.wait_for_tts_to_finish()
            tts.shutdown_tts()
        except:
            # If interrupted during cleanup, just shutdown
            try:
                tts.shutdown_tts()
            except:
                pass
    
    # Exit with proper code for signal termination
    sys.exit(128 + signum)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='TalkiTo - Speak command output using TTS',
        usage='%(prog)s [options] <command> [arguments...]\n       %(prog)s claude'
    )
    
    # Basic options
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('--auto-skip-tts', action='store_true', 
                        help='Automatically skip to newer content when TTS is behind')
    parser.add_argument('--disable-tts', action='store_true',
                        help='Disable TTS output completely')
    parser.add_argument('--profile', type=str, 
                        help='Use a specific profile (claude, aider, etc.)')
    
    # Verbosity levels
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Increase verbosity (can be used multiple times: -v, -vv, -vvv)')
    parser.add_argument('--verbosity', type=int, metavar='LEVEL',
                        help='Set verbosity level directly (0-3)')
    
    # TTS options
    tts_group = parser.add_argument_group('TTS options')
    tts_group.add_argument('--tts-provider', type=str, 
                           choices=['system', 'openai', 'aws', 'polly', 'azure', 'gcloud', 'elevenlabs', 'deepgram'],
                           help='TTS provider to use')
    tts_group.add_argument('--tts-voice', type=str, 
                           help='Voice to use (provider-specific)')
    tts_group.add_argument('--tts-region', type=str, 
                           help='AWS/Azure region for cloud TTS')
    tts_group.add_argument('--tts-language', type=str, default='en-US',
                           help='Language code for TTS (default: en-US)')
    tts_group.add_argument('--tts-rate', type=float,
                           help='Speech rate (provider-specific, typically 0.5-2.0)')
    tts_group.add_argument('--tts-pitch', type=float,
                           help='Speech pitch (provider-specific)')
    tts_group.add_argument('--capture-tts-output', type=str, metavar='FILE',
                           help='Capture TTS output to file instead of playing')
    
    # ASR options
    asr_group = parser.add_argument_group('ASR options')
    asr_group.add_argument('--asr-mode', type=str, 
                           choices=['off', 'auto-input', 'continuous', 'tap-to-talk'],
                           default='auto-input',
                           help='ASR mode (default: auto-input)')
    asr_group.add_argument('--asr-provider', type=str,
                           choices=['google', 'gcloud', 'assemblyai', 'deepgram', 'houndify', 'aws', 'bing'],
                           help='ASR provider to use')
    asr_group.add_argument('--asr-language', type=str, default='en-US',
                           help='Language code for ASR (default: en-US)')
    asr_group.add_argument('--asr-model', type=str,
                           help='ASR model to use (provider-specific)')
    
    # Communication options
    comm_group = parser.add_argument_group('Communication options')
    comm_group.add_argument('--sms-recipients', type=str,
                           help='SMS recipient phone numbers (comma-separated)')
    comm_group.add_argument('--whatsapp-recipients', type=str,
                           help='WhatsApp recipient phone numbers (comma-separated)')
    comm_group.add_argument('--slack-channel', type=str,
                           help='Slack channel to send messages to')
    comm_group.add_argument('--webhook-port', type=int, default=8080,
                           help='Port for webhook server (default: 8080)')
    
    # Debug options
    debug_group = parser.add_argument_group('Debug options')
    debug_group.add_argument('--record', type=str, metavar='FILE',
                           help='Record session to file for replay')
    debug_group.add_argument('--replay', type=str, metavar='FILE',
                           help='Replay a recorded session file')
    debug_group.add_argument('--no-output', action='store_true',
                           help='Hide output during replay')
    debug_group.add_argument('--log-file', type=str, metavar='FILE',
                           help='Enable logging and write to specified file')
    
    # MCP server mode
    parser.add_argument('--mcp-server', action='store_true',
                        help='Run as MCP (Model Context Protocol) server')
    parser.add_argument('--mcp-sse-server', action='store_true',
                        help='Run as MCP server with SSE transport for real-time notifications')
    parser.add_argument('--port', type=int, metavar='PORT',
                        help='Port to run the MCP SSE server on (default: auto-find from 8000)')
    parser.add_argument('--disable-mcp', action='store_true',
                        help='Disable MCP server when running claude command (use wrapper mode only)')
    
    # Setup helpers
    parser.add_argument('--setup-slack', action='store_true',
                        help='Show instructions for setting up Slack bot')
    parser.add_argument('--setup-whatsapp', action='store_true',
                        help='Show instructions for setting up WhatsApp with Twilio')
    
    # Update command
    parser.add_argument('--update', action='store_true',
                        help='Check for updates and install the latest version')
    parser.add_argument('--force-update', action='store_true',
                        help='Force update even if already up to date')
    
    # Command and arguments
    parser.add_argument('command', nargs='?', 
                        help='Command to run')
    parser.add_argument('arguments', nargs=argparse.REMAINDER,
                        help='Arguments for the command')
    
    args = parser.parse_args()
    
    # Handle verbosity levels
    if args.verbosity is not None:
        args.verbose = args.verbosity
    
    # Handle special commands
    if args.command == "init" and args.arguments and args.arguments[0] == "claude":
        # This is the 'init claude' command
        args.init_claude = True
        return args
    
    # Handle setup helpers
    if args.setup_slack:
        args.show_slack_setup = True
        return args
    
    if args.setup_whatsapp:
        args.show_whatsapp_setup = True
        return args
    
    # Handle update command
    if args.update or args.force_update:
        args.show_update = True
        return args
    
    # Validate arguments
    if not args.replay and not args.command and not args.mcp_server and not args.mcp_sse_server and not args.setup_slack and not args.setup_whatsapp and not args.update and not args.force_update:
        parser.error('Command is required unless using --replay, --mcp-server, --mcp-sse-server, --setup-slack, --setup-whatsapp, --update or init claude')
    
    # Check if any command arguments look like talkito options
    if args.command and args.arguments:
        talkito_options = {
            '--log-file', '--tts-provider', '--asr-provider', '--tts-voice', '--tts-region', '--tts-language',
            '--tts-rate', '--tts-pitch', '--capture-tts-output', '--asr-mode', '--asr-language', '--asr-model',
            '--sms-recipients', '--whatsapp-recipients', '--slack-channel', '--webhook-port', '--record', '--replay',
            '--no-output', '--port', '--disable-mcp', '--auto-skip-tts', '--disable-tts', '--profile', '--verbosity',
            '-v', '--verbose', '--mcp-server', '--mcp-sse-server', '--setup-slack', '--setup-whatsapp'
        }
        
        for i, arg in enumerate(args.arguments):
            if arg in talkito_options:
                print(f"\nError: Talkito option '{arg}' was placed after the command '{args.command}'", file=sys.stderr)
                print("\nDid you mean: talkito " + (arg + " ... " + args.command + " " + ' '.join(args.arguments[:i]) + " " + ' '.join(args.arguments[i+2:])).strip() + "?", file=sys.stderr)
                sys.exit(1)
    
    return args


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
    _ = get_shared_state()  # This ensures the state singleton is initialized and loaded
    
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


async def run_claude_wrapper(args) -> int:
    cmd = [args.command] + args.arguments

    # Build kwargs for run_with_talkito
    kwargs = {
        'verbosity': args.verbose,
        'asr_mode': args.asr_mode,
        'record_file': args.record,
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
    from .core import run_with_talkito
    return await run_with_talkito(cmd, **kwargs)

async def run_claude_with_sse(args) -> int:
    """Run Claude with SSE MCP server support with fallback mechanisms"""
    import subprocess
    import time

    from .mcp import find_available_port
    
    # Use specified port or find an available one
    if args.port:
        port = args.port
    else:
        port = find_available_port(8000)
    address = "http://127.0.0.1:"
    sse_process = None
    claude_process = None
    
    # Helper to kill processes
    def kill_process(proc):
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    
    try:
        # Step A: Run talkito --mcp-sse-server in background
        print(f"Starting talkito MCP (SSE) server on port {port}...")
        # Use the installed talkito command if available, otherwise use -m
        import shutil
        talkito_cmd = shutil.which('talkito')
        if talkito_cmd:
            sse_cmd = [talkito_cmd, "--mcp-sse-server", "--port", str(port)]
        else:
            # Fallback to module execution
            sse_cmd = [sys.executable, "-m", "talkito", "--mcp-sse-server", "--port", str(port)]
        
        # Add log file if specified
        if args.log_file:
            sse_cmd.extend(["--log-file", args.log_file + ".sse"])
        
        # Add TTS/ASR provider arguments if specified
        if args.tts_provider:
            sse_cmd.extend(["--tts-provider", args.tts_provider])
        if args.asr_provider:
            sse_cmd.extend(["--asr-provider", args.asr_provider])
        
        # Pass the client command to indicate this is for Claude
        if args.command:
            sse_cmd.extend(["--client-command", args.command])

        sse_process = subprocess.Popen(
            sse_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )
        
        # Wait a bit for the server to start
        time.sleep(2)
        
        # Check if server started successfully
        if sse_process.poll() is not None:
            # Read error output
            stdout, stderr = sse_process.communicate()
            error_msg = "SSE server failed to start"
            if stderr:
                error_msg += f"\nError: {stderr.decode('utf-8', errors='replace')}"
            if stdout:
                error_msg += f"\nOutput: {stdout.decode('utf-8', errors='replace')}"
            raise Exception(error_msg)
        
        # Step C: Run init_claude with SSE configuration
        from .claude_init import init_claude
        if not init_claude(transport="sse", address=address, port=port):
            raise Exception("Failed to configure Claude for SSE")
        
        # Step D: Show configuration status
        print_configuration_status(args)
        
        # Step E: Run Claude
        print("Starting Claude...")
        claude_cmd = ["claude"] + args.arguments
        claude_process = subprocess.run(claude_cmd)
        
        return claude_process.returncode
        
    except Exception as e:
        print(f"SSE mode failed: {e}", file=sys.stderr)
        
        # Step E: Fallback to stdio mode
        print("Falling back to stdio mode...")
        kill_process(sse_process)
        
        try:
            from .claude_init import init_claude
            if not init_claude(transport="stdio"):
                raise Exception("Failed to configure Claude for stdio")
            
            # Try Claude again with stdio
            claude_cmd = ["claude"] + args.arguments
            claude_process = subprocess.run(claude_cmd)
            return claude_process.returncode
            
        except Exception as e2:
            print(f"Stdio mode failed: {e2}", file=sys.stderr)
            
            # Step F: Fall back to traditional talkito wrapper
            print("Falling back to traditional talkito wrapper...")
            return await run_claude_wrapper(args)
    
    finally:
        # Clean up any remaining processes
        kill_process(sse_process)


async def run_claude_hybrid(args) -> int:
    """Run Claude with in-process MCP server and wrapper functionality"""
    import threading
    import time
    import signal
    from .mcp import app, find_available_port
    from .claude_init import init_claude
    from .tts import stop_tts_immediately
    
    # Find available port
    port = args.port if args.port else find_available_port(8000)
    if not port:
        print("Error: Could not find an available port", file=sys.stderr)
        return 1
    
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
            
            # Apply configuration
            mcp.configure_mcp_server(**config)
            
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
                # Run the server (this blocks)
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
                if stderr_content:
                    print(f"MCP server stderr output:\n{stderr_content}", file=sys.stderr)
                
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
        print(f"[Main] Configuring Claude to use MCP server at port {port}", file=sys.stderr)
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


async def run_talkito_command(args) -> int:
    """Run talkito with the given arguments"""
    global core_instance
    
    # Special handling for 'claude' command
    if args.command == 'claude':
        # Check if MCP is disabled
        if args.disable_mcp:
            # Use wrapper mode only
            return await run_claude_wrapper(args)
        else:
            # Use hybrid approach
            return await run_claude_hybrid(args)
    
    # Build command list
    cmd = [args.command] + args.arguments
    
    # Build kwargs for run_with_talkito
    kwargs = {
        'verbosity': args.verbose,
        'asr_mode': args.asr_mode,
        'record_file': args.record,
    }
    
    # Add log file if specified
    if args.log_file:
        kwargs['log_file'] = args.log_file
    
    # Add profile if specified
    if args.profile:
        kwargs['profile'] = args.profile
    elif args.command:
        # Try to auto-detect profile based on command
        kwargs['profile'] = os.path.basename(args.command)
    
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
    from .core import run_with_talkito
    return await run_with_talkito(cmd, **kwargs)


async def replay_session(args) -> Union[int, List[Tuple[float, str, int]]]:
    """Replay a recorded session"""
    from .core import replay_recorded_session
    
    # Build TTS config if specified
    tts_config = build_tts_config(args)
    
    # Use the command argument as profile name if provided
    command_name = args.command if args.command else None
    
    # Run the replay using the original function signature
    result = await replay_recorded_session(
        args.replay,
        auto_skip_tts=args.auto_skip_tts,
        tts_config=tts_config,
        record_file=args.record,
        capture_tts=args.capture_tts_output is not None,
        disable_tts=args.disable_tts,
        show_output=not args.no_output,
        command_name=command_name,
        verbosity=args.verbose,
        log_file=args.log_file
    )
    
    # Handle capture_tts output
    if args.capture_tts_output and isinstance(result, list):
        # Save captured TTS output to file
        try:
            with open(args.capture_tts_output, 'w', encoding='utf-8') as f:
                for timestamp, text, line_number in result:
                    f.write(text + '\n')
            print(f"\nCaptured {len(result)} TTS lines to: {args.capture_tts_output}", file=sys.stderr)
        except Exception as e:
            print(f"Error writing TTS capture file: {e}", file=sys.stderr)
        return 0
    else:
        # Normal exit with exit code
        return result


def run_mcp_server():
    """Run the MCP server"""
    try:
        from .mcp import main as mcp_main
        # Pass through the original sys.argv to the MCP server
        # but remove the --mcp-server argument itself
        original_argv = sys.argv[:]
        sys.argv = [sys.argv[0]] + [arg for arg in sys.argv[1:] if arg != '--mcp-server']
        mcp_main()
    except ImportError:
        print("Error: MCP SDK not found. Install with: pip install mcp", file=sys.stderr)
        sys.exit(1)


def run_mcp_sse_server():
    """Run the MCP server with SSE transport"""
    try:
        from .mcp import main as mcp_main
        # Pass through the original sys.argv to the MCP server
        # but remove the --mcp-sse-server argument and add --transport sse
        original_argv = sys.argv[:]
        sys.argv = [sys.argv[0]] + ['--transport', 'sse'] + [arg for arg in sys.argv[1:] if arg != '--mcp-sse-server']
        print(f"[DEBUG] Running MCP server with SSE transport, args: {sys.argv}", file=sys.stderr)
        mcp_main()
    except ImportError as e:
        print(f"Error: Failed to import MCP server: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        print("Make sure the MCP SDK is installed: pip install mcp", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error running MCP server: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def show_slack_setup():
    """Show instructions for setting up Slack bot"""
    from .templates import SLACK_BOT_MANIFEST
    import json
    import shutil
    
    print("🚀 Talkito Slack Bot Setup Instructions")
    print("=" * 60)
    print()
    print("Follow these steps to set up your Slack bot for Talkito:")
    print()
    print("1. Create a new Slack app:")
    print("   • Go to https://api.slack.com/apps")
    print("   • Click 'Create New App'")
    print("   • Choose 'From an app manifest'")
    print("   • Select your workspace")
    print("   • Paste the manifest below when prompted")
    print()
    print("2. Install the app to your workspace:")
    print("   • After creating the app, go to 'Install App' in the sidebar")
    print("   • Click 'Install to Workspace'")
    print("   • Authorize the requested permissions")
    print()
    print("3. Get your tokens:")
    print("   • Bot User OAuth Token: Settings → Install App → Bot User OAuth Token")
    print("   • App-Level Token: Settings → Basic Information → App-Level Tokens")
    print("     - Click 'Generate Token and Scopes'")
    print("     - Name: 'Socket Mode'")
    print("     - Add scope: 'connections:write'")
    print("     - Click 'Generate'")
    print()
    print("4. Set environment variables:")
    print("   export SLACK_BOT_TOKEN='xoxb-...'  # Bot User OAuth Token")
    print("   export SLACK_APP_TOKEN='xapp-...'  # App-Level Token")
    print("   export SLACK_CHANNEL='#talkito-comms'  # Your default channel")
    print()
    print("=" * 60)
    print("SLACK APP MANIFEST:")
    print("=" * 60)
    print()
    
    # Pretty print the manifest
    try:
        manifest_dict = json.loads(SLACK_BOT_MANIFEST)
        manifest_pretty = json.dumps(manifest_dict, indent=2)
        print(manifest_pretty)
    except:
        print(SLACK_BOT_MANIFEST)
    
    print()
    print("=" * 60)
    
    # Check if pbcopy is available on macOS
    if shutil.which('pbcopy'):
        try:
            import subprocess
            subprocess.run(['pbcopy'], input=SLACK_BOT_MANIFEST.encode(), check=True)
            print("✅ Manifest copied to clipboard! (macOS)")
        except:
            pass
    #
    # print()
    # print("After setup, test with:")
    # print("  talkito --slack-channel '#your-channel' echo 'Hello Slack!'")
    # print()
    # print("Or use with Claude:")
    # print("  talkito claude")
    # print("  Then: /talkito:start_slack_mode")
    # print()


def show_whatsapp_setup():
    """Show instructions for setting up WhatsApp with Twilio"""
    print("📱 Talkito WhatsApp Setup Instructions")
    print("=" * 60)
    print()
    print("Follow these steps to set up WhatsApp messaging for Talkito:")
    print()
    print("1. Create a Twilio account:")
    print("   • Sign up at https://www.twilio.com/")
    print("   • Note your Account SID and Auth Token from the dashboard")
    print()
    print("2. Set up WhatsApp Sandbox (for testing):")
    print("   • Go to https://console.twilio.com/us1/develop/sms/try-it-out/whatsapp-learn")
    print("   • Note the Twilio WhatsApp number (usually +1 415 523 8886)")
    print("   • Send the join code shown (e.g., 'join <code>') to the WhatsApp number")
    print("   • NOTE you will need to re send that join <code> very 24 hours")
    print("   • You'll receive a confirmation message")
    print()
    print("3. Configure Zrok webhook:")
    print("   • Install zrok: https://github.com/openziti/zrok/releases/latest")
    print("   • Enable zrok: zrok enable <token> (get token from https://zrok.io)")
    print("   • Reserve a share: zrok reserve public http://localhost:8080")
    print("   • Note the reserved token (e.g., 'es5hi3nzrstm')")
    print("   • This token MUST be set as ZROK_RESERVED_TOKEN for WhatsApp to work")
    print("   • Set webhook URL in Twilio Console:")
    print("     - Go to Messaging → Settings → WhatsApp Sandbox Settings")
    print("     - Set 'When a message comes in' to: https://<token>.share.zrok.io/whatsapp")
    print("     - Method: POST")
    print()
    print("4. Set environment variables:")
    print("   export TWILIO_ACCOUNT_SID='ACxxxxxx'          # Your Account SID")
    print("   export TWILIO_AUTH_TOKEN='xxxxxx'             # Your Auth Token")
    print("   export TWILIO_WHATSAPP_NUMBER='+14155238886'  # Twilio's WhatsApp number")
    print("   export WHATSAPP_RECIPIENTS='+1234567890'      # Your WhatsApp number")
    print("   export ZROK_RESERVED_TOKEN='xxxxx'            # Your zrok token (see step 4)")
    print()
    print("5. Test the setup:")
    print("   # Send a test message")
    print("   talkito --whatsapp-recipients '+1234567890' echo 'Hello WhatsApp!'")
    print()
    print("   # Or use with Claude")
    print("   talkito claude")
    print("   Then: /talkito:start_whatsapp_mode")
    print()
    print("=" * 60)
    print("IMPORTANT NOTES:")
    print("=" * 60)
    print()
    print("• For production use, upgrade to a Twilio WhatsApp Business API account")
    print("• The sandbox is limited to approved contacts who have joined")
    print("• Messages expire after 24 hours of inactivity in the sandbox")
    print("• Reply to WhatsApp messages to keep the session active")
    print()
    print("For detailed webhook setup instructions, see WEBHOOK_SETUP.md")
    print()


async def main_async() -> int:
    """Async main function"""
    args = parse_arguments()
    
    # Handle MCP server modes synchronously before entering async context
    if args.mcp_server or args.mcp_sse_server:
        # MCP servers have their own event loops, so we can't run them from within asyncio.run()
        # This is handled in main() instead
        return 0
    
    try:
        if args.replay:
            return await replay_session(args)
        else:
            return await run_talkito_command(args)
    except KeyboardInterrupt:
        # Clean up quickly without waiting
        if asr:
            try:
                asr.stop_dictation()
            except:
                pass
        try:
            tts.shutdown_tts()  # Shutdown immediately without waiting
        except:
            pass
        return 130  # Standard exit code for Ctrl+C
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        # Clean up only if not interrupted
        if 'KeyboardInterrupt' not in str(type(sys.exc_info()[1])):
            if asr:
                try:
                    asr.stop_dictation()
                except:
                    pass
            # Wait for TTS to finish before shutting down
            try:
                tts.wait_for_tts_to_finish()
                tts.shutdown_tts()
            except KeyboardInterrupt:
                # If interrupted during cleanup, just shutdown immediately
                try:
                    tts.shutdown_tts()
                except:
                    pass


def main():
    """Main entry point for the CLI"""
    # Check and apply any staged updates first
    from .update import check_and_apply_staged_update
    check_and_apply_staged_update()
    
    # Handle MCP server mode before entering asyncio context
    args = parse_arguments()
    
    # Handle special commands that don't need async
    if hasattr(args, 'init_claude') and args.init_claude:
        from .claude_init import init_claude
        success = init_claude()
        sys.exit(0 if success else 1)
    
    if hasattr(args, 'show_slack_setup') and args.show_slack_setup:
        show_slack_setup()
        sys.exit(0)
    
    if hasattr(args, 'show_whatsapp_setup') and args.show_whatsapp_setup:
        show_whatsapp_setup()
        sys.exit(0)
    
    if hasattr(args, 'show_update') and args.show_update:
        from .update import TalkitoUpdater
        updater = TalkitoUpdater()
        success = updater.update(force=args.force_update)
        sys.exit(0 if success else 1)
    
    if args.mcp_server:
        run_mcp_server()
        sys.exit(0)
    
    if args.mcp_sse_server:
        run_mcp_sse_server()
        sys.exit(0)
    
    # Run everything else in asyncio
    exit_code = asyncio.run(main_async())
    sys.exit(exit_code)


if __name__ == '__main__':
    main()