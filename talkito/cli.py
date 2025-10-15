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

# ruff: noqa: E402

# Apply huggingface_hub timeout patch FIRST before any other imports that might use it
from . import models  # noqa: F401

# Suppress deprecation warnings early
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="websockets")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="uvicorn")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="spacy")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="click")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="weasel")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Try to load .env files if available
try:
    from dotenv import load_dotenv, set_key
    # Load .env first (takes precedence)
    load_dotenv()
    # Also load .talkito.env (won't override existing vars from .env)
    load_dotenv('.talkito.env')
    dotenv_available = True
except ImportError:
    # python-dotenv not installed, continue without it
    dotenv_available = False

import os
import argparse
import asyncio
import json
import platform
import signal
import shutil
import subprocess
import sys
import termios
import traceback
import tty

from typing import List, Optional, Union, Tuple

# Import from our package
from . import __version__
from . import asr
from . import tts
from .core import replay_recorded_session, run_with_talkito, signal_handler, TalkitoCore
from .claude import init_claude, run_claude_extensions
from .logs import log_message, setup_logging
from .mcp import main as mcp_main
from .state import get_shared_state, get_status_summary, initialize_providers_early, show_tap_to_talk_notification_once
from .templates import SLACK_BOT_MANIFEST
from .update import check_and_apply_staged_update, TalkitoUpdater

# Check Python version
if sys.version_info < (3, 8):
    print("Error: Python 3.8 or higher required", file=sys.stderr)
    print("Your version:", sys.version, file=sys.stderr)
    sys.exit(1)

# Global references for signal handlers
core_instance: Optional[TalkitoCore] = None

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='TalkiTo - Speak command output using TTS',
        usage='%(prog)s [options] <command> [arguments...]\n       %(prog)s claude'
    )
    
    # Basic options
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('--dont-auto-skip-tts', action='store_true', 
                        help='Disable automatic skipping to newer content when TTS is behind')
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
                           choices=['system', 'openai', 'aws', 'polly', 'azure', 'gcloud', 'elevenlabs', 'deepgram', 'kittentts', 'kokoro'],
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
    tts_group.add_argument('--tts-mode', type=str, 
                           choices=['off', 'full', 'auto-skip'],
                           default='auto-skip',
                           help='TTS mode (default: auto-skip)')
    
    # ASR options
    asr_group = parser.add_argument_group('ASR options')
    asr_group.add_argument('--asr-mode', type=str,
                           default='tap-to-talk',
                           help='ASR mode: off, auto-input, tap-to-talk, or file:<path> for testing (default: tap-to-talk)')
    asr_group.add_argument('--asr-provider', type=str,
                           choices=['google', 'gcloud', 'assemblyai', 'deepgram', 'houndify', 'aws', 'bing', 'local_whisper'],
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
    if args.verbose > 0 and args.verbosity is None:
        args.verbosity = args.verbose
    elif args.verbosity is None:
        args.verbosity = 0
    
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

    if not args.profile and args.command:
        args.profile = os.path.basename(args.command)
    
    # Validate arguments - show welcome screen if no command provided
    if not args.replay and not args.command and not args.mcp_server and not args.mcp_sse_server and not args.setup_slack and not args.setup_whatsapp and not args.update and not args.force_update:
        args.show_welcome = True
        return args
    
    # Check if any command arguments look like talkito options
    if args.command and args.arguments:
        talkito_options = {
            '--log-file', '--tts-provider', '--asr-provider', '--tts-voice', '--tts-region', '--tts-language',
            '--tts-rate', '--tts-pitch', '--tts-mode', '--capture-tts-output', '--asr-mode', '--asr-language', '--asr-model',
            '--sms-recipients', '--whatsapp-recipients', '--slack-channel', '--webhook-port', '--record', '--replay',
            '--no-output', '--port', '--disable-mcp', '--dont-auto-skip-tts', '--disable-tts', '--profile', '--verbosity',
            '-v', '--verbose', '--mcp-server', '--mcp-sse-server', '--setup-slack', '--setup-whatsapp'
        }
        
        for i, arg in enumerate(args.arguments):
            if arg in talkito_options:
                print(f"\nError: Talkito option '{arg}' was placed after the command '{args.command}'", file=sys.stderr)
                print("\nDid you mean: talkito " + (arg + " ... " + args.command + " " + ' '.join(args.arguments[:i]) + " " + ' '.join(args.arguments[i+2:])).strip() + "?", file=sys.stderr)
                sys.exit(1)
    
    return args


def print_configuration_status(args):
    """Print the current TTS/ASR and communication configuration"""

    # Initialize providers early (triggers availability checks and download prompts)
    initialize_providers_early(args)
    
    # Handle backwards compatibility for TTS mode
    tts_mode = args.tts_mode
    if args.disable_tts:
        tts_mode = 'off'
    elif args.dont_auto_skip_tts:
        tts_mode = 'full'
    
    # Set ASR and TTS modes in shared state so they're available for status display
    shared_state = get_shared_state()
    shared_state.asr_mode = args.asr_mode
    shared_state.tts_mode = tts_mode
    
    # Show one-time notification about tap-to-talk change if needed
    show_tap_to_talk_notification_once()
    
    # Don't pass configured providers to allow showing actual working providers after fallback
    status = get_status_summary(
        tts_override=True, 
        asr_override=(args.asr_mode != "off")
    )

    # Print with the same format but add the note about .talkito.env
    print(f"╭ {status}")

async def run_talkito_command(args) -> int:
    """Run talkito with the given arguments"""
    global core_instance

    # Set environment variables for provider preferences
    # Check if the TTS provider is valid first, fall back to system on macOS if not
    if args.tts_provider:
        # Validate the provider before setting it
        log_message("INFO", f"About to validate TTS provider: {args.tts_provider}")
        provider_valid = tts.validate_provider_config(args.tts_provider)
        log_message("INFO", f"TTS provider validation completed - valid: {provider_valid}")
        if not provider_valid:
            if platform.system() == 'Darwin':  # macOS
                print("Falling back to system TTS provider on macOS")
                args.tts_provider = 'system'
                os.environ['TALKITO_PREFERRED_TTS_PROVIDER'] = 'system'
            else:
                # On non-macOS systems, let it fail as before
                os.environ['TALKITO_PREFERRED_TTS_PROVIDER'] = args.tts_provider
        else:
            os.environ['TALKITO_PREFERRED_TTS_PROVIDER'] = args.tts_provider

    # If no ASR provider specified on command line, check environment variable
    if not args.asr_provider:
        env_asr_provider = os.environ.get('TALKITO_PREFERRED_ASR_PROVIDER')
        if env_asr_provider:
            args.asr_provider = env_asr_provider
            log_message("INFO", f"Using ASR provider from environment: {env_asr_provider}")

    if args.asr_provider:
        os.environ['TALKITO_PREFERRED_ASR_PROVIDER'] = args.asr_provider

    # Initialize providers early (triggers availability checks and download prompts)
    initialize_providers_early(args)

    # Handle backwards compatibility for TTS mode
    tts_mode = args.tts_mode
    if args.disable_tts:
        tts_mode = 'off'
    elif args.dont_auto_skip_tts:
        tts_mode = 'full'

    # Set ASR and TTS modes in shared state so they're available for status display
    shared_state = get_shared_state()

    # Parse ASR mode - check for file: prefix
    asr_mode_value = args.asr_mode
    if asr_mode_value.startswith('file:'):
        # Extract file path and optional delay: file:path or file:path:delay
        parts = asr_mode_value[5:].split(':', 1)  # Remove 'file:' prefix and split
        shared_state.asr_source_file = parts[0]
        shared_state.asr_file_delay = float(parts[1]) if len(parts) > 1 else 0.1
        shared_state.asr_mode = 'file'
    else:
        shared_state.asr_mode = asr_mode_value
        shared_state.asr_source_file = None
        shared_state.asr_file_delay = 0.1

    shared_state.tts_mode = tts_mode

    # Special handling for 'claude' command
    if args.command == 'claude':
        log_message("INFO", "Starting claude mode")
        try:
            # Check if MCP is disabled
            log_message("DEBUG", f"Starting claude mode args.disable_mcp {args.disable_mcp}")
            if not args.disable_mcp:
                await run_claude_extensions(args)
        finally:
            pass  # Signal handling moved to core.py

    # Build command list
    cmd = [args.command] + args.arguments

    # Use the high-level API from core
    return await run_with_talkito(cmd, args)


async def replay_session(args) -> Union[int, List[Tuple[float, str, int]]]:
    """Replay a recorded session"""

    # Use the command argument as profile name if provided
    command_name = args.command if args.command else None

    # Run the replay using the original function signature
    result = await replay_recorded_session(args = args, command_name=command_name)

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
        # Pass through the original sys.argv to the MCP server
        # but remove the --mcp-server argument itself
        sys.argv = [sys.argv[0]] + [arg for arg in sys.argv[1:] if arg != '--mcp-server']
        mcp_main()
    except ImportError:
        print("Error: MCP SDK not found. Install with: pip install mcp", file=sys.stderr)
        sys.exit(1)


def run_mcp_sse_server():
    """Run the MCP server with SSE transport"""
    try:
        # Pass through the original sys.argv to the MCP server
        # but remove the --mcp-sse-server argument and add --transport sse
        sys.argv = [sys.argv[0]] + ['--transport', 'sse'] + [arg for arg in sys.argv[1:] if arg != '--mcp-sse-server']
        print(f"[DEBUG] Running MCP server with SSE transport, args: {sys.argv}", file=sys.stderr)
        mcp_main()
    except ImportError as e:
        print(f"Error: Failed to import MCP server: {e}", file=sys.stderr)
        traceback.print_exc()
        print("Make sure the MCP SDK is installed: pip install mcp", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error running MCP server: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


def show_slack_setup():
    """Show instructions for setting up Slack bot"""

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
    except Exception:
        print(SLACK_BOT_MANIFEST)

    print()
    print("=" * 60)

    # Check if pbcopy is available on macOS
    if shutil.which('pbcopy'):
        try:
            subprocess.run(['pbcopy'], input=SLACK_BOT_MANIFEST.encode(), check=True)
            print("✅ Manifest copied to clipboard! (macOS)")
        except Exception:
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


def show_welcome_and_config():
    """Show welcome screen and configuration menu when talkito is run without a command"""

    # ASCII Header
    print("""
╭─────────────────────────────────────────────────────────────╮
│   ████████╗ █████╗ ██╗     ██╗  ██╗██╗████████╗ ██████╗     │
│   ╚══██╔══╝██╔══██╗██║     ██║ ██╔╝██║╚══██╔══╝██╔═══██╗    │
│      ██║   ███████║██║     █████╔╝ ██║   ██║   ██║   ██║    │
│      ██║   ██╔══██║██║     ██╔═██╗ ██║   ██║   ██║   ██║    │
│      ██║   ██║  ██║███████╗██║  ██╗██║   ██║   ╚██████╔╝    │
│      ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝   ╚═╝    ╚═════╝     │
╰─────────────────────────────────────────────────────────────╯""")

    print("\n🎯 Welcome to TalkiTo! Let's configure your preferred providers.\n")

    # Check current environment variables
    current_tts = os.environ.get('TALKITO_PREFERRED_TTS_PROVIDER', 'auto')
    current_asr = os.environ.get('TALKITO_PREFERRED_ASR_PROVIDER', 'auto')

    print(f"Current preferences:\n  TTS Provider: {current_tts}\n  ASR Provider: {current_asr}\n")

    # Check available providers
    try:
        asr_available = True
    except ImportError:
        asr_available = False

    accessible_tts = tts.check_tts_provider_accessibility()
    if asr_available:
        accessible_asr = asr.check_asr_provider_accessibility()
    else:
        accessible_asr = {}

    # TTS Configuration with interactive menu
    tts_choices = ['system', 'openai', 'aws', 'polly', 'azure', 'gcloud', 'elevenlabs', 'deepgram', 'kittentts', 'kokoro', 'auto']
    tts_available = []

    for provider in tts_choices:
        if provider == 'auto':
            tts_available.append((provider, True, "Let TalkiTo choose automatically"))
        else:
            available = accessible_tts.get(provider, {}).get('available', False)
            note = accessible_tts.get(provider, {}).get('note', '')
            tts_available.append((provider, available, note))

    new_tts_provider = show_interactive_menu(
        "TTS Provider",
        tts_available,
        current_tts,
        test_tts_provider if 'test_tts_provider' in globals() else None
    )

    # ASR Configuration with interactive menu
    new_asr_provider = None
    if asr_available:
        asr_choices = ['google', 'gcloud', 'assemblyai', 'deepgram', 'houndify', 'aws', 'bing', 'local_whisper', 'auto']
        asr_available_list = []

        for provider in asr_choices:
            if provider == 'auto':
                asr_available_list.append((provider, True, "Let TalkiTo choose automatically"))
            else:
                available = accessible_asr.get(provider, {}).get('available', False)
                note = accessible_asr.get(provider, {}).get('note', '')
                asr_available_list.append((provider, available, note))

        new_asr_provider = show_interactive_menu(
            "ASR Provider",
            asr_available_list,
            current_asr,
            None  # No test function for ASR providers
        )

    # Save preferences
    config_file = '.talkito.env'
    changes_made = False

    if new_tts_provider is not None and new_tts_provider != current_tts:
        if dotenv_available:
            if new_tts_provider and new_tts_provider != 'auto':
                set_key(config_file, 'TALKITO_PREFERRED_TTS_PROVIDER', new_tts_provider, quote_mode="never")
                print(f"✅ Set TTS provider to: {new_tts_provider}")
            else:
                # Remove the key for auto selection
                env_content = ""
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        env_content = f.read()
                    # Remove the line containing TALKITO_PREFERRED_TTS_PROVIDER
                    lines = env_content.split('\n')
                    new_lines = [line for line in lines if not line.startswith('TALKITO_PREFERRED_TTS_PROVIDER=')]
                    with open(config_file, 'w') as f:
                        f.write('\n'.join(new_lines))
                print("✅ Set TTS provider to: auto")
            changes_made = True
        else:
            print("⚠️  To save preferences, install python-dotenv: pip install python-dotenv")
            if new_tts_provider and new_tts_provider != 'auto':
                print(f"   For now, set manually: export TALKITO_PREFERRED_TTS_PROVIDER={new_tts_provider}")
            else:
                print("   For now, unset manually: unset TALKITO_PREFERRED_TTS_PROVIDER")

    if asr_available and new_asr_provider is not None and new_asr_provider != current_asr:
        if dotenv_available:
            if new_asr_provider and new_asr_provider != 'auto':
                set_key(config_file, 'TALKITO_PREFERRED_ASR_PROVIDER', new_asr_provider, quote_mode="never")
                print(f"✅ Set ASR provider to: {new_asr_provider}")
            else:
                # Remove the key for auto selection
                env_content = ""
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        env_content = f.read()
                    # Remove the line containing TALKITO_PREFERRED_ASR_PROVIDER
                    lines = env_content.split('\n')
                    new_lines = [line for line in lines if not line.startswith('TALKITO_PREFERRED_ASR_PROVIDER=')]
                    with open(config_file, 'w') as f:
                        f.write('\n'.join(new_lines))
                print("✅ Set ASR provider to: auto")
            changes_made = True
        else:
            if new_asr_provider and new_asr_provider != 'auto':
                print(f"   For now, set manually: export TALKITO_PREFERRED_ASR_PROVIDER={new_asr_provider}")
            else:
                print("   For now, unset manually: unset TALKITO_PREFERRED_ASR_PROVIDER")

    if changes_made:
        print(f"\n💾 Preferences saved to {config_file}")

    print("\n" + "─" * 65)
    print("🚀 Quick Start Examples:")
    print("  talkito echo 'Hello World!'          # Basic TTS demo")
    print("  talkito --asr-mode auto-input claude # Voice-enabled Claude")
    print("  talkito --setup-slack                # Setup Slack integration")
    print("  talkito --setup-whatsapp             # Setup WhatsApp integration")
    print("\n📚 For more info: talkito --help")
    print("─" * 65)


def show_interactive_menu(title, options, current_choice, test_function=None):
    """Show an interactive menu with arrow key navigation"""

    def get_key():
        """Get a single keypress"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
            if key == '\x1b':  # ESC sequence
                key += sys.stdin.read(2)
            return key
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    # Find current selection index
    selected = 0
    for i, (provider, available, note) in enumerate(options):
        if provider == current_choice:
            selected = i
            break

    # Display menu header
    print(f"\n═══ {title} Configuration ═══")
    print(f"{title} selection (Use ↑↓ arrows, Enter to select, 't' to test, 'q' to quit):")

    # Calculate how many lines the menu will take
    menu_lines = len(options) + 2  # options + controls line + spacing

    # Display initial menu
    for i, (provider, available, note) in enumerate(options):
        cursor = "➤ " if i == selected else "  "
        status = "✅" if available else "❌"
        current_marker = " (current)" if provider == current_choice else ""
        test_marker = " [Press 't' to test]" if i == selected and available and provider != 'auto' and test_function else ""
        print(f"{cursor}{provider:<15} {status} {note}{current_marker}{test_marker}")

    print("\nControls: ↑/↓ = navigate, Enter = select, 't' = test, 'q' = quit")

    while True:
        # Get user input
        key = get_key()

        if key == '\x1b[A':  # Up arrow
            selected = (selected - 1) % len(options)
            # Move cursor up to start of menu and redraw
            print(f'\033[{menu_lines}A', end='')  # Move up
            for i, (provider, available, note) in enumerate(options):
                cursor = "➤ " if i == selected else "  "
                status = "✅" if available else "❌"
                current_marker = " (current)" if provider == current_choice else ""
                test_marker = " [Press 't' to test]" if i == selected and available and provider != 'auto' and test_function else ""
                print(f"\033[2K{cursor}{provider:<15} {status} {note}{current_marker}{test_marker}")
            print("\033[2K\nControls: ↑/↓ = navigate, Enter = select, 't' = test, 'q' = quit")

        elif key == '\x1b[B':  # Down arrow
            selected = (selected + 1) % len(options)
            # Move cursor up to start of menu and redraw
            print(f'\033[{menu_lines}A', end='')  # Move up
            for i, (provider, available, note) in enumerate(options):
                cursor = "➤ " if i == selected else "  "
                status = "✅" if available else "❌"
                current_marker = " (current)" if provider == current_choice else ""
                test_marker = " [Press 't' to test]" if i == selected and available and provider != 'auto' and test_function else ""
                print(f"\033[2K{cursor}{provider:<15} {status} {note}{current_marker}{test_marker}")
            print("\033[2K\nControls: ↑/↓ = navigate, Enter = select, 't' = test, 'q' = quit")

        elif key == '\x03':  # Ctrl+C
            print()  # Just move to next line
            return None
        elif key == '\r' or key == '\n':  # Enter
            provider = options[selected][0]
            available = options[selected][1]
            if available:
                print()  # Just move to next line
                return provider
            else:
                print(f"\n❌ {provider} is not available. Please choose an available provider.")

        elif key == 't' or key == 'T':  # Test
            provider = options[selected][0]
            available = options[selected][1]
            if available and provider != 'auto' and test_function:
                # Just run the test without screen changes
                try:
                    test_function(provider)
                except Exception:
                    pass  # Test function handles its own output
            # Do nothing - no screen changes, just continue
        elif key == 'q' or key == 'Q':  # Quit
            print()  # Just move to next line
            return None


def test_tts_provider(provider):
    """Test a TTS provider by speaking a sample phrase"""
    test_text = f"Hello! This is {provider} text-to-speech."

    if provider == 'system':
        # Test system TTS
        if platform.system() == 'Darwin':  # macOS
            subprocess.run(['say', test_text], check=True)
        else:
            # Try espeak, festival, or flite
            for cmd in ['espeak', 'festival', 'flite']:
                if subprocess.run(['which', cmd], capture_output=True).returncode == 0:
                    if cmd == 'espeak':
                        subprocess.run(['espeak', test_text], check=True)
                    elif cmd == 'festival':
                        subprocess.run(['festival', '--tts'], input=test_text.encode(), check=True)
                    elif cmd == 'flite':
                        subprocess.run(['flite', '-t', test_text], check=True)
                    break
    else:
        # Test other providers using talkito's TTS system with worker

        # Set temporary environment variable for provider
        old_provider = os.environ.get('TALKITO_PREFERRED_TTS_PROVIDER', None)
        shared_state = get_shared_state()
        old_state_provider = shared_state.tts_provider

        try:
            os.environ['TALKITO_PREFERRED_TTS_PROVIDER'] = provider
            shared_state.set_tts_config(provider=provider)

            # For local models, ensure model is preloaded before testing
            if provider in ['kittentts', 'kokoro']:
                tts.preload_local_model(provider)

            # Use direct provider approach to avoid worker synchronization issues
            provider_instance = tts.create_tts_provider(provider)
            if provider_instance:
                success = provider_instance.speak(test_text, use_process_control=False)
                if not success:
                    raise Exception("TTS synthesis failed")
            else:
                raise Exception("Could not create TTS provider")

        finally:
            pass  # No worker cleanup needed

            # Restore environment and shared state
            if old_provider is not None:
                os.environ['TALKITO_PREFERRED_TTS_PROVIDER'] = old_provider
            elif 'TALKITO_PREFERRED_TTS_PROVIDER' in os.environ:
                del os.environ['TALKITO_PREFERRED_TTS_PROVIDER']

            # Restore shared state
            shared_state.set_tts_config(provider=old_state_provider)


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
            except Exception:
                pass
        try:
            tts.shutdown_tts()  # Shutdown immediately without waiting
        except Exception:
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
                except Exception:
                    pass
            # Wait for TTS to finish before shutting down
            try:
                tts.wait_for_tts_to_finish()
                tts.shutdown_tts()
            except KeyboardInterrupt:
                # If interrupted during cleanup, just shutdown immediately
                try:
                    tts.shutdown_tts()
                except Exception:
                    pass


def main():
    """Main entry point for the CLI"""
    # Install signal handlers early to handle hangs during initialization
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Check and apply any staged updates first
    check_and_apply_staged_update()

    # Handle MCP server mode before entering asyncio context
    args = parse_arguments()

    # Set up logging early if log file specified (before any other operations)
    if args.log_file:
        setup_logging(args.log_file, mode='w')  # Use 'w' for fresh log on startup

    # Handle special commands that don't need async
    if hasattr(args, 'init_claude') and args.init_claude:
        success = init_claude()
        sys.exit(0 if success else 1)

    if hasattr(args, 'show_slack_setup') and args.show_slack_setup:
        show_slack_setup()
        sys.exit(0)

    if hasattr(args, 'show_whatsapp_setup') and args.show_whatsapp_setup:
        show_whatsapp_setup()
        sys.exit(0)

    if hasattr(args, 'show_update') and args.show_update:
        updater = TalkitoUpdater()
        success = updater.update(force=args.force_update)
        sys.exit(0 if success else 1)

    if hasattr(args, 'show_welcome') and args.show_welcome:
        show_welcome_and_config()
        sys.exit(0)

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