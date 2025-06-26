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

import sys
import os
import argparse
import signal
import asyncio
from typing import List, Optional, Union, Tuple
from pathlib import Path

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
    # Wait for TTS to finish before shutting down
    tts.wait_for_tts_to_finish()
    tts.shutdown_tts()
    
    # Exit with proper code for signal termination
    sys.exit(128 + signum)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='TalkiTo - Speak command output using TTS',
        usage='%(prog)s [options] <command> [arguments...]\n       %(prog)s init claude'
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
                           choices=['system', 'openai', 'polly', 'azure', 'gcloud', 'elevenlabs', 'deepgram'],
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
    
    # Validate arguments
    if not args.replay and not args.command and not args.mcp_server:
        parser.error('Command is required unless using --replay or --mcp-server')
    
    return args


def build_tts_config(args) -> dict:
    """Build TTS configuration from command line arguments"""
    config = {}
    
    if args.tts_provider:
        config['provider'] = args.tts_provider
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


async def run_talkito_command(args) -> int:
    """Run talkito with the given arguments"""
    global core_instance
    
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
        from .mcp_server import main as mcp_main
        mcp_main()
    except ImportError:
        print("Error: MCP SDK not found. Install with: pip install mcp", file=sys.stderr)
        sys.exit(1)


async def main_async() -> int:
    """Async main function"""
    args = parse_arguments()
    
    # Handle MCP server mode synchronously before entering async context
    if args.mcp_server:
        # MCP server has its own event loop, so we can't run it from within asyncio.run()
        # This is handled in main() instead
        return 0
    
    try:
        if args.replay:
            return await replay_session(args)
        else:
            return await run_talkito_command(args)
    except KeyboardInterrupt:
        return 130  # Standard exit code for Ctrl+C
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        # All cleanup is now handled in run_command
        pass


def main():
    """Main entry point for the CLI"""
    # Handle MCP server mode before entering asyncio context
    args = parse_arguments()
    
    # Handle special commands that don't need async
    if hasattr(args, 'init_claude') and args.init_claude:
        from .claude_init import init_claude
        success = init_claude()
        sys.exit(0 if success else 1)
    
    if args.mcp_server:
        run_mcp_server()
        sys.exit(0)
    
    # Run everything else in asyncio
    exit_code = asyncio.run(main_async())
    sys.exit(exit_code)


if __name__ == '__main__':
    main()