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

"""Communication providers for remote interaction with talkito - supports Twilio SMS, WhatsApp, and Slack."""

import os
import re
import time
import hashlib
import threading
import signal
import subprocess
import sys
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Tuple, Deque
from queue import Queue, Empty
from collections import deque
from difflib import SequenceMatcher

# Import centralized logging utilities
from .logs import log_message

# Import shared state for tool use detection
from .state import get_shared_state

# Optional imports for providers
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

try:
    from slack_sdk import WebClient as SlackClient
    from slack_sdk.errors import SlackApiError
    from slack_sdk.socket_mode import SocketModeClient
    from slack_sdk.socket_mode.response import SocketModeResponse
    from slack_sdk.socket_mode.request import SocketModeRequest
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False


@dataclass
class Message:
    """Represents a message to send or received"""
    content: str
    sender: str  # Phone number, Slack user ID, etc.
    channel: str  # SMS, WhatsApp, Slack channel
    timestamp: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    message_id: Optional[str] = None
    reply_to: Optional[str] = None  # For threading


@dataclass 
class CommsConfig:
    """Configuration for communication providers"""
    # Twilio
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_phone_number: Optional[str] = None
    twilio_whatsapp_number: Optional[str] = None
    
    # Slack
    slack_bot_token: Optional[str] = None
    slack_app_token: Optional[str] = None
    slack_channel: Optional[str] = None
    
    # General settings
    webhook_port: int = 8080
    webhook_host: str = "0.0.0.0"
    webhook_use_tunnel: bool = True  # Auto-start zrok
    zrok_reserved_token: Optional[str] = None  # Optional zrok reserved share token for stable URL
    max_message_length: int = 1600  # SMS limit
    batch_delay: float = 0.5  # Delay between batched messages
    
    # Recipients
    sms_recipients: List[str] = field(default_factory=list)
    whatsapp_recipients: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Load from environment variables if not provided"""
        self.twilio_account_sid = self.twilio_account_sid or os.environ.get('TWILIO_ACCOUNT_SID')
        self.twilio_auth_token = self.twilio_auth_token or os.environ.get('TWILIO_AUTH_TOKEN')
        self.twilio_phone_number = self.twilio_phone_number or os.environ.get('TWILIO_PHONE_NUMBER')
        self.twilio_whatsapp_number = self.twilio_whatsapp_number or os.environ.get('TWILIO_WHATSAPP_NUMBER')
        self.slack_bot_token = self.slack_bot_token or os.environ.get('SLACK_BOT_TOKEN')
        self.slack_app_token = self.slack_app_token or os.environ.get('SLACK_APP_TOKEN')
        self.slack_channel = self.slack_channel or os.environ.get('SLACK_CHANNEL')
        self.zrok_reserved_token = self.zrok_reserved_token or os.environ.get('ZROK_RESERVED_TOKEN')


class CommsProvider(ABC):
    """Abstract base class for communication providers"""

    def __init__(self, config: CommsConfig):
        self.config = config
        self.active = True

    @abstractmethod
    def send_message(self, message: Message) -> bool:
        """Send a message through the provider"""
        pass

    @abstractmethod
    def start(self, input_callback: Callable[[Message], None]):
        """Start the provider and set up message receiving"""
        pass

    @abstractmethod
    def stop(self):
        """Stop the provider"""
        pass

    def format_output(self, text: str) -> List[str]:
        """Format terminal output for messaging, splitting if needed"""
        # Import inside function to avoid circular dependency (core imports comms at module level)
        from .core import ANSI_PATTERN

        # Remove ANSI codes using comprehensive pattern from core.py
        text = ANSI_PATTERN.sub('', text)
        
        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        
        if not text:
            return []
        
        # Split into chunks if too long
        chunks = []
        max_len = self.config.max_message_length
        
        if len(text) <= max_len:
            return [text]
        
        # Try to split on newlines first
        lines = text.split('\n')
        current_chunk = []
        current_length = 0
        
        for line in lines:
            line_len = len(line) + 1  # +1 for newline
            if current_length + line_len > max_len and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_length = line_len
            else:
                current_chunk.append(line)
                current_length += line_len
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks


class TwilioBaseProvider(CommsProvider):
    """Base class for Twilio providers with common functionality"""
    
    def __init__(self, config: CommsConfig, phone_number_config_key: str, provider_name: str):
        super().__init__(config)
        if not TWILIO_AVAILABLE:
            raise ImportError("Twilio library not installed. Run: pip install twilio")
        
        # Get phone number from config
        phone_number = getattr(config, phone_number_config_key)
        if not all([config.twilio_account_sid, config.twilio_auth_token, phone_number]):
            raise ValueError(f"Twilio {provider_name} credentials not configured")
        
        # Suppress Twilio HTTP logging when no logger is set
        self._suppress_twilio_logging()
        
        self.client = TwilioClient(config.twilio_account_sid, config.twilio_auth_token)
        self.phone_number = phone_number
        log_message("INFO", f"Initialized {provider_name} provider with number {phone_number}")
    
    def _suppress_twilio_logging(self):
        """Suppress verbose Twilio HTTP logging."""
        # Always suppress verbose Twilio HTTP logging
        import logging as stdlib_logging
        twilio_logger = stdlib_logging.getLogger('twilio.http_client')
        twilio_logger.setLevel(stdlib_logging.WARNING)
    
    def start(self, input_callback: Callable[[Message], None]):
        """Start webhook server for receiving messages."""
        log_message("INFO", f"{self.__class__.__name__} started")
    
    def stop(self):
        """Stop the provider."""
        self.active = False
        log_message("INFO", f"{self.__class__.__name__} stopped")


class TwilioSMSProvider(TwilioBaseProvider):
    """Twilio SMS provider"""
    
    def __init__(self, config: CommsConfig):
        super().__init__(config, 'twilio_phone_number', 'SMS')
    
    def send_message(self, message: Message) -> bool:
        """Send SMS message"""
        try:
            chunks = self.format_output(message.content)
            
            for i, chunk in enumerate(chunks):
                if i > 0:
                    time.sleep(self.config.batch_delay)
                
                # Add chunk indicator if multiple
                if len(chunks) > 1:
                    chunk = f"[{i+1}/{len(chunks)}] {chunk}"
                
                self.client.messages.create(
                    body=chunk,
                    from_=self.phone_number,
                    to=message.sender
                )
            
            return True
        except Exception as e:
            log_message("ERROR", f"Failed to send SMS: {e}")
            return False


class TwilioWhatsAppProvider(TwilioBaseProvider):
    """WhatsApp provider via Twilio"""
    
    def __init__(self, config: CommsConfig):
        super().__init__(config, 'twilio_whatsapp_number', 'WhatsApp')
    
    def send_message(self, message: Message) -> bool:
        """Send WhatsApp message"""
        try:
            chunks = self.format_output(message.content)
            for i, chunk in enumerate(chunks):
                if i > 0:
                    time.sleep(self.config.batch_delay)
                
                # WhatsApp numbers need 'whatsapp:' prefix
                from_number = f"whatsapp:{self.phone_number}"
                to_number = f"whatsapp:{message.sender}" if not message.sender.startswith("whatsapp:") else message.sender
                
                self.client.messages.create(
                    body=chunk,
                    from_=from_number,
                    to=to_number
                )
            
            return True
        except Exception as e:
            log_message("ERROR", f"Failed to send WhatsApp message: {e}")
            return False
    


class SlackProvider(CommsProvider):
    """Slack provider"""
    
    def __init__(self, config: CommsConfig):
        super().__init__(config)
        if not SLACK_AVAILABLE:
            raise ImportError("Slack SDK not installed. Run: pip install slack-sdk")
        
        if not config.slack_bot_token:
            raise ValueError("Slack bot token not configured")
        
        self.client = SlackClient(token=config.slack_bot_token)
        self.channel = config.slack_channel
        self.socket_client = None
        self.input_callback = None
        self.connected_event = threading.Event()  # Event to signal when connected
        self.rate_limited = False  # Track rate limit status
        self.rate_limit_reset_time = None  # When rate limit resets
        
        # Enable Socket Mode if app token is provided
        if config.slack_app_token:
            self.socket_client = SocketModeClient(
                app_token=config.slack_app_token,
                web_client=self.client
            )
            log_message("INFO", f"Initialized Slack provider with Socket Mode for channel {self.channel}")
        else:
            log_message("INFO", f"Initialized Slack provider for channel {self.channel} (send-only)")

    def prep_for_slack(self, s: str) -> str:
        # minimal, safe unescape for common cases
        return (s.replace('\\r\\n', '\n')
                .replace('\\n', '\n')
                .replace('\\r', '\n')
                .replace('\\t', '\t'))

    def send_message(self, message: Message) -> bool:
        """Send Slack message."""
        log_message("DEBUG", f"SlackProvider.send_message called with: content='{message.content[:50]}...', sender={message.sender}, channel={message.channel}")
        
        # Check if provider is active
        if not self.active:
            log_message("DEBUG", "[COMMS] Provider is not active, returning False")
            return False
        
        # Check if we're currently rate limited
        if self.rate_limited:
            if self.rate_limit_reset_time and time.time() < self.rate_limit_reset_time:
                log_message("WARNING", f"Slack is rate limited until {time.ctime(self.rate_limit_reset_time)}")
                return False
            else:
                # Rate limit period has passed
                self.rate_limited = False
                self.rate_limit_reset_time = None
                log_message("INFO", "Slack rate limit has been lifted")
        
        try:
            content = self.prep_for_slack(message.content)
            
            # Determine target channel
            target = message.sender if message.sender != self.channel else self.channel
            log_message("DEBUG", f"Target channel: {target}, self.channel: {self.channel}")
            
            log_message("INFO", f"Sending Slack message to {target}: {content[:50]}...")
            try:
                result = self.client.chat_postMessage(
                    channel=target,
                    text=content,
                    thread_ts=message.reply_to
                )
            except Exception as api_e:
                log_message("DEBUG" ,f"[COMMS DEBUG] Exception during chat_postMessage: {type(api_e).__name__}: {str(api_e)}")
                print(f"error {type(api_e).__name__}: {str(api_e)}")
                raise
            
            # Store message timestamp for threading
            if result.get("ok"):
                message.message_id = result["ts"]
                log_message("DEBUG", f"Slack message sent successfully, ts={result['ts']}")
            else:
                log_message("DEBUG", f"Slack API returned ok=False: {result}")
            
            return True
        except SlackApiError as e:
            error_response = e.response
            log_message("DEBUG", f"SlackApiError: {error_response}")

            error_code = error_response.get('error', '')
            if error_code == 'not_in_channel':
                log_message("ERROR", f"Bot is not in channel {target}. Please invite the bot to the channel with /invite @talkito")
                self._last_error = f"not_in_channel: {target}"
                return False
            elif error_code == 'channel_not_found':
                log_message("ERROR", f"Channel {target} not found. Please create the channel or use an existing one.")
                self._last_error = f"channel_not_found: {target}"
                return False
            elif error_code == 'ratelimited':
                # Extract retry_after if available
                retry_after = error_response.get('retry_after', 60)  # Default to 60 seconds
                self.rate_limited = True
                self.rate_limit_reset_time = time.time() + retry_after
                log_message("ERROR", f"Slack rate limited! Will retry after {retry_after} seconds")
                log_message("ERROR", f"Temporarily disabling Slack until {time.ctime(self.rate_limit_reset_time)}")
                # Optionally disable the provider temporarily
                self.active = False
                # Schedule re-enabling
                threading.Timer(retry_after, self._re_enable_after_rate_limit).start()
                return False
            else:
                log_message("DEBUG", f"Failed to send Slack message: {e}")
                log_message("ERROR", f"Error details: {error_response}")
                return False
        except Exception as e:
            log_message("ERROR", f"Unexpected error in SlackProvider.send_message: {type(e).__name__}: {str(e)}")
            traceback.print_exc(file=sys.stderr)
            log_message("ERROR", f"Traceback: {traceback.format_exc()}")
            return False
    
    def _re_enable_after_rate_limit(self):
        """Re-enable the provider after rate limit period."""
        self.active = True
        self.rate_limited = False
        self.rate_limit_reset_time = None
        log_message("INFO", "Slack provider re-enabled after rate limit period")
    
    def _process_slack_event(self, client: SocketModeClient, req: SocketModeRequest):
        """Process incoming Slack events."""
        if req.type == "events_api":
            # Acknowledge the request
            response = SocketModeResponse(envelope_id=req.envelope_id)
            client.send_socket_mode_response(response)
            
            # Handle different event types
            event = req.payload.get("event", {})
            event_type = event.get("type")
            
            if event_type == "message":
                # Ignore bot messages to prevent loops
                if event.get("bot_id"):
                    return
                
                # Extract message details
                text = event.get("text", "")
                user = event.get("user", "")
                channel = event.get("channel", "")
                thread_ts = event.get("thread_ts")
                
                # Create Message object
                message = Message(
                    content=text,
                    sender=channel,  # Use channel as sender
                    channel="slack",
                    reply_to=thread_ts
                )
                
                # Call the input callback
                if self.input_callback:
                    self.input_callback(message)
                    
                log_message("DEBUG", f"Received Slack message from {user} in {channel}: {text[:50]}...")
    
    def start(self, input_callback: Callable[[Message], None]):
        """Start Slack event listener."""
        self.input_callback = input_callback
        
        if self.socket_client:
            # Register event handler
            self.socket_client.socket_mode_request_listeners.append(self._process_slack_event)
            
            # Start Socket Mode client in a separate thread
            def run_socket_mode():
                try:
                    self.socket_client.connect()
                    log_message("INFO", "Slack Socket Mode connected")
                    self.connected_event.set()  # Signal that we're connected
                    # Keep the connection alive
                    while self.active:
                        time.sleep(1)
                except Exception as e:
                    log_message("ERROR", f"Slack Socket Mode error: {e}")
                    self.connected_event.set()  # Also set on error to unblock waiters
            
            socket_thread = threading.Thread(target=run_socket_mode, daemon=True)
            socket_thread.start()
            log_message("INFO", "Slack provider started with Socket Mode")
        else:
            self.connected_event.set()  # No async connection needed for send-only mode
            log_message("INFO", "Slack provider started (send-only mode)")
    
    def wait_for_connection(self, timeout: float = 5.0) -> bool:
        """Wait for connection to be established."""
        return self.connected_event.wait(timeout)
    
    def stop(self):
        """Stop the provider."""
        self.active = False
        if self.socket_client:
            try:
                self.socket_client.close()
                log_message("INFO", "Slack Socket Mode disconnected")
            except Exception:
                pass
        log_message("INFO", "Slack provider stopped")


class CommunicationManager:
    """Manages all communication providers and message routing"""

    # Cache configuration constants (matching tts.py pattern)
    CACHE_SIZE = 1000  # Cache size for similarity checking
    SIMILARITY_THRESHOLD = 0.85  # How similar text must be to be considered a repeat

    # Buffer-specific constants
    BUFFER_CACHE_SIZE = 100  # Cache size for buffer deduplication
    BUFFER_SIMILARITY_THRESHOLD = 0.90  # Higher threshold for buffer content
    BUFFER_MAX_LINES = 100000  # Maximum lines to keep in buffer
    
    def __init__(self, config: CommsConfig):
        self.config = config
        self.providers: List[CommsProvider] = []
        self.input_queue: Queue[Message] = Queue()
        self.output_queue: Queue[Message] = Queue()
        self.tunnel_process = None
        self.webhook_url = None
        self.active = False
        self.worker_thread = None
        self.current_session_id = self._generate_session_id()
        
        # Message cache to prevent duplicate sends
        self.sent_cache: Deque[Tuple[str, str, float]] = deque(maxlen=self.CACHE_SIZE)  # (content, channel, timestamp)
        self._cache_lock = threading.Lock()
        
        # Buffer management for slack context
        self.processed_lines_buffer: List[str] = []  # Global buffer for processed but unspoken lines
        self.buffer_cache: Deque[Tuple[str, str, float]] = deque(maxlen=self.BUFFER_CACHE_SIZE)  # (buffer_content, channel, timestamp)
        self._buffer_lock = threading.Lock()
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]
    
    def _is_duplicate_message(self, content: str, channel: str) -> bool:
        """Check if message was recently sent to the same channel"""
        with self._cache_lock:
            # Check for exact matches first
            for cached_content, cached_channel, _ in self.sent_cache:
                if cached_content == content and cached_channel == channel:
                    log_message("DEBUG", f"Duplicate message detected (exact match): '{content[:50]}...' to {channel}")
                    return True

            # Check for similarity
            for cached_content, cached_channel, _ in self.sent_cache:
                if cached_channel == channel:
                    similarity = SequenceMatcher(None, content.lower(), cached_content.lower()).ratio()
                    if similarity >= self.SIMILARITY_THRESHOLD:
                        log_message("INFO", f"Message '{content[:50]}...' is {similarity:.2%} similar to '{cached_content[:50]}...' on {channel}")
                        return True

        return False

    def _add_to_cache(self, content: str, channel: str):
        """Add message to sent cache"""
        with self._cache_lock:
            self.sent_cache.append((content, channel, time.time()))

    def _check_and_cache_message(self, content: str, channel: str) -> bool:
        """Atomically check for duplicate and add to cache if not duplicate. Returns True if message should be sent, False if duplicate."""
        current_time = time.time()

        with self._cache_lock:
            # Check for exact matches first
            for cached_content, cached_channel, _ in self.sent_cache:
                if cached_content == content and cached_channel == channel:
                    log_message("DEBUG", f"Duplicate message detected (exact match): '{content[:50]}...' to {channel}")
                    return False

            # Check for similarity
            for cached_content, cached_channel, _ in self.sent_cache:
                if cached_channel == channel:
                    similarity = SequenceMatcher(None, content.lower(), cached_content.lower()).ratio()
                    if similarity >= self.SIMILARITY_THRESHOLD:
                        log_message("INFO", f"Message '{content[:50]}...' is {similarity:.2%} similar to '{cached_content[:50]}...' on {channel}")
                        return False

            # Not a duplicate - add to cache while still holding the lock
            self.sent_cache.append((content, channel, current_time))
            return True
    
    def reset_message_cache(self):
        """Reset message cache - useful for testing"""
        with self._cache_lock:
            self.sent_cache.clear()
        log_message("INFO", "Communication message cache reset")
    
    def _is_duplicate_buffer(self, buffer_lines: List[str], channel: str) -> bool:
        """Check if buffer content is a duplicate"""
        if not buffer_lines:
            return False

        buffer_content = '\n'.join(buffer_lines)

        with self._buffer_lock:
            # Check for exact matches first
            for cached_content, cached_channel, _ in self.buffer_cache:
                if cached_channel == channel and cached_content == buffer_content:
                    return True

            # Check for similarity
            for cached_content, cached_channel, _ in self.buffer_cache:
                if cached_channel == channel:
                    similarity = SequenceMatcher(None, buffer_content.lower(), cached_content.lower()).ratio()
                    if similarity >= self.BUFFER_SIMILARITY_THRESHOLD:
                        log_message("INFO", f"Buffer content is {similarity:.2%} similar to cached buffer on {channel}")
                        return True

        return False
    
    def _add_buffer_to_cache(self, buffer_lines: List[str], channel: str):
        """Add buffer content to cache"""
        if not buffer_lines:
            return
        buffer_content = '\n'.join(buffer_lines)
        with self._buffer_lock:
            self.buffer_cache.append((buffer_content, channel, time.time()))
    
    def add_to_buffer(self, line: str, active_profile, verbosity_level: int = 2):
        """Add a line to the processed lines buffer with verbosity filtering"""
        if not line or not line.strip():
            return

        # Check if this line should be filtered based on verbosity level
        if active_profile.should_skip(line, verbosity_level):
            log_message("DEBUG", f"Filtered buffer line by profile (verbosity={verbosity_level}): '{line[:50]}...'")
            return

        cleaned_line = line.replace(" │ │", "").strip().replace("│", "")
        
        if cleaned_line and any(c.isalnum() for c in cleaned_line):
            log_message("DEBUG", f"Added to buffer ({cleaned_line})")
            self.processed_lines_buffer.append(cleaned_line)
            # Keep buffer size reasonable
            if len(self.processed_lines_buffer) > self.BUFFER_MAX_LINES:
                self.processed_lines_buffer = self.processed_lines_buffer[-self.BUFFER_MAX_LINES:]
    
    def clear_buffer(self):
        """Clear the processed lines buffer"""
        self.processed_lines_buffer.clear()
    
    def add_provider(self, provider_type: str, recipients: List[str] = None) -> bool:
        """Add a communication provider - returns True if successful, False otherwise."""
        log_message("INFO", f"Adding {provider_type} provider")
        
        try:
            if provider_type == "sms" and TWILIO_AVAILABLE:
                provider = TwilioSMSProvider(self.config)
                self.providers.append(provider)
                log_message("INFO", "Added SMS provider")
                return True
            
            elif provider_type == "whatsapp" and TWILIO_AVAILABLE:
                # Check if ZROK_RESERVED_TOKEN is set (required for WhatsApp)
                zrok_token = os.environ.get('ZROK_RESERVED_TOKEN')
                log_message("DEBUG", f"ZROK_RESERVED_TOKEN check: {'set' if zrok_token else 'not set'}")
                
                if not zrok_token:
                    log_message("ERROR", "ZROK_RESERVED_TOKEN is required for WhatsApp support")
                    print("\n❌ WhatsApp requires ZROK_RESERVED_TOKEN to be set!")
                    print("   Run 'talkito --setup-whatsapp' for setup instructions")
                    return False
                
                log_message("DEBUG", f"Creating TwilioWhatsAppProvider with config: account_sid={'***' if self.config.twilio_account_sid else 'None'}, auth_token={'***' if self.config.twilio_auth_token else 'None'}, whatsapp_number={self.config.twilio_whatsapp_number}")
                provider = TwilioWhatsAppProvider(self.config)
                self.providers.append(provider)
                log_message("INFO", "Added WhatsApp provider")
                return True
            
            elif provider_type == "slack" and SLACK_AVAILABLE:
                log_message("DEBUG", f"Creating SlackProvider with config: bot_token={'***' if self.config.slack_bot_token else 'None'}, app_token={'***' if self.config.slack_app_token else 'None'}, channel={self.config.slack_channel}")
                
                # Check if required credentials are available
                if not self.config.slack_bot_token:
                    log_message("ERROR", "Cannot add Slack provider: SLACK_BOT_TOKEN not set")
                    return False
                if not self.config.slack_app_token:
                    log_message("ERROR", "Cannot add Slack provider: SLACK_APP_TOKEN not set")
                    return False
                    
                provider = SlackProvider(self.config)
                self.providers.append(provider)
                log_message("INFO", "Added Slack provider")
                return True
            
            else:
                log_message("WARNING", f"Provider {provider_type} not available - TWILIO_AVAILABLE={TWILIO_AVAILABLE}, SLACK_AVAILABLE={SLACK_AVAILABLE}")
                return False
        
        except Exception as e:
            log_message("ERROR", f"Failed to add provider {provider_type}: {e}")
            log_message("ERROR", f"Traceback: {traceback.format_exc()}")
            return False
    
    def start(self):
        """Start all providers and worker thread"""
        self.active = True
        
        # Start providers
        for provider in self.providers:
            provider.start(self._handle_input_message)

        # Start webhook server if needed
        if any(isinstance(p, (TwilioSMSProvider, TwilioWhatsAppProvider)) for p in self.providers):
            self._start_webhook_server()

        # Start worker thread
        self.worker_thread = threading.Thread(target=self._output_worker, daemon=True)
        self.worker_thread.start()

        log_message("INFO", f"Communication manager started with session {self.current_session_id}")
    
    def stop(self):
        """Stop all providers"""
        self.active = False
        
        for provider in self.providers:
            provider.stop()
        
        # Unregister webhook endpoints from API server
        from .api import get_api_server
        api_server = get_api_server()
        if api_server.running:
            api_server.unregister_endpoint('/sms')
            api_server.unregister_endpoint('/whatsapp')
            log_message("INFO", "Webhook endpoints unregistered")
        
        if self.tunnel_process:
            try:
                # Kill the entire process group to ensure zrok and any children are terminated
                if os.name != 'nt':
                    os.killpg(os.getpgid(self.tunnel_process.pid), signal.SIGTERM)
                else:
                    self.tunnel_process.terminate()
                self.tunnel_process.wait(timeout=5)
                log_message("INFO", "Zrok tunnel disconnected")
            except Exception:
                try:
                    if os.name != 'nt':
                        os.killpg(os.getpgid(self.tunnel_process.pid), signal.SIGKILL)
                    else:
                        self.tunnel_process.kill()
                except Exception:
                    pass
        
        log_message("INFO", "Communication manager stopped")
    
    def send_output(self, text: str, recipients: List[str] = None, processed_lines_buffer: List[str] = None):
        """Queue output to be sent to all configured recipients"""
        if not text.strip():
            return
        
        # Check shared state for active modes
        shared_state = get_shared_state()


        # Send to SMS recipients
        sms_recipients = recipients or self.config.sms_recipients
        for phone in sms_recipients:
            # Check for duplicate before queuing (skip if in tool use mode)
            channel_key = f"sms:{phone}"
            in_tool_use = shared_state.get_in_tool_use()
            
            if in_tool_use and "Do you " in text:
                msg = Message(
                    content=text,
                    sender=phone,
                    channel="sms",
                    session_id=self.current_session_id
                )
                self.output_queue.put(msg)
                self._add_to_cache(text, channel_key)
            elif self._check_and_cache_message(text, channel_key):
                msg = Message(
                    content=text,
                    sender=phone,
                    channel="sms",
                    session_id=self.current_session_id
                )
                self.output_queue.put(msg)
            else:
                log_message("INFO", f"Recently sent to SMS {phone}: '{text[:50]}...'")
        
        # Send to WhatsApp recipients (include mode recipient if active)
        whatsapp_recipients = list(self.config.whatsapp_recipients)
        if shared_state.whatsapp_mode_active and shared_state.communication.whatsapp_to_number:
            if shared_state.communication.whatsapp_to_number not in whatsapp_recipients:
                whatsapp_recipients.append(shared_state.communication.whatsapp_to_number)

        for phone in whatsapp_recipients:
            # Check for duplicate before queuing (skip if in tool use mode)
            channel_key = f"whatsapp:{phone}"
            in_tool_use = shared_state.get_in_tool_use()
            
            if in_tool_use:
                msg = Message(
                    content=text,
                    sender=phone,
                    channel="whatsapp",
                    session_id=self.current_session_id
                )
                self.output_queue.put(msg)
                self._add_to_cache(text, channel_key)
            elif self._check_and_cache_message(text, channel_key):
                msg = Message(
                    content=text,
                    sender=phone,
                    channel="whatsapp",
                    session_id=self.current_session_id
                )
                self.output_queue.put(msg)
            else:
                log_message("INFO", f"Recently sent to WhatsApp {phone}: '{text[:50]}...'")
        
        # Send to Slack (use mode channel if active)
        if any(isinstance(p, SlackProvider) for p in self.providers):
            # Prepare final text with buffer if in slack mode
            final_text = text.replace("│", "").strip()
            buffer_lines_to_use = processed_lines_buffer or self.processed_lines_buffer

            if self.config.slack_channel and buffer_lines_to_use:
                # Check if buffer content is duplicate
                if not self._is_duplicate_buffer(buffer_lines_to_use, f"slack:{self.config.slack_channel}"):
                    # Prepend buffered lines as code block for slack
                    buffer_text = '\n'.join(buffer_lines_to_use)
                    buffer_text = buffer_text.replace('```', '')  # Escape existing code blocks
                    final_text = f"```\n{buffer_text}\n```\n{final_text}"
                    log_message("DEBUG", f"Prepended {len(buffer_lines_to_use)} buffered lines for slack")
                    
                    # Add buffer to cache
                    self._add_buffer_to_cache(buffer_lines_to_use, f"slack:{self.config.slack_channel}")
                else:
                    log_message("INFO", f"Skipped duplicate buffer content for Slack {self.config.slack_channel}")
                
                # Clear buffer after using (either our internal buffer or passed buffer)
                if processed_lines_buffer:
                    processed_lines_buffer.clear()
                else:
                    self.clear_buffer()
            
            # Check for duplicate message before queuing (skip if in tool use mode)
            channel_key = f"slack:{self.config.slack_channel}"
            in_tool_use = shared_state.get_in_tool_use()
            
            if in_tool_use:
                log_message("DEBUG", f"sending final text {len(final_text)} {final_text} to slack")
                msg = Message(
                    content=final_text,
                    sender=self.config.slack_channel,
                    channel="slack",
                    session_id=self.current_session_id
                )
                self.output_queue.put(msg)
                self._add_to_cache(final_text, channel_key)
            elif self._check_and_cache_message(final_text, channel_key):
                log_message("DEBUG", f"sending final text {len(final_text)} {final_text} to slack")
                msg = Message(
                    content=final_text,
                    sender=self.config.slack_channel,
                    channel="slack",
                    session_id=self.current_session_id
                )
                self.output_queue.put(msg)
            else:
                log_message("INFO", f"Recently sent to Slack {self.config.slack_channel}: '{final_text[:50]}...'")
    
    def get_input(self, timeout: float = None) -> Optional[str]:
        """Get input from communication channels."""
        try:
            # Use non-blocking get to avoid freezing the terminal
            if timeout == 0 or timeout is None:
                message = self.input_queue.get_nowait()
            else:
                message = self.input_queue.get(timeout=timeout)
            return message.content
        except Empty:
            return None
    
    def get_input_message(self, timeout: float = None) -> Optional[Message]:
        """Get input message object from communication channels."""
        try:
            # Use non-blocking get to avoid freezing the terminal
            if timeout == 0 or timeout is None:
                message = self.input_queue.get_nowait()
            else:
                message = self.input_queue.get(timeout=timeout)
            return message
        except Empty:
            return None
    
    def get_active_provider_types(self) -> List[str]:
        """Get list of active provider types."""
        active_types = []
        for provider in self.providers:
            if isinstance(provider, TwilioSMSProvider):
                active_types.append("sms")
            elif isinstance(provider, TwilioWhatsAppProvider):
                active_types.append("whatsapp")
            elif isinstance(provider, SlackProvider):
                active_types.append("slack")
        return active_types
    
    def _handle_input_message(self, message: Message):
        """Handle incoming message from providers"""
        # Associate with current session
        message.session_id = self.current_session_id
        self.input_queue.put(message)
        log_message("INFO", f"Received input from {message.sender}: {message.content}")
    
    def _output_worker(self):
        """Worker thread to process output queue."""
        # Map channel types to provider classes
        channel_to_provider = {
            "sms": TwilioSMSProvider,
            "whatsapp": TwilioWhatsAppProvider,
            "slack": SlackProvider
        }
        
        while self.active:
            try:
                message = self.output_queue.get(timeout=1)
                
                # Find appropriate provider
                provider_class = channel_to_provider.get(message.channel)
                if provider_class:
                    for provider in self.providers:
                        if isinstance(provider, provider_class):
                            provider.send_message(message)
                            break
                
            except Empty:
                continue
            except Exception as e:
                log_message("ERROR", f"Error in output worker: {e}")
    
    def _start_webhook_server(self):
        """Register webhook endpoints with the API server"""
        from .api import get_api_server
        
        # Get the API server instance
        api_server = get_api_server()
        
        if not api_server.running:
            log_message("ERROR", "API server is not running. Cannot register webhook endpoints.")
            return
        
        # Create handler functions for each endpoint
        parent_self = self
        
        def handle_sms_webhook(data: dict) -> dict:
            """Handle SMS webhook requests"""
            from_number = data.get('From', '')
            body = data.get('Body', '')
            message_sid = data.get('MessageSid', '')  # Twilio's unique message ID
            
            log_message("INFO", f"[WEBHOOK] Parsed SMS message from {from_number}: {body}")
            
            # Create and handle the message
            message = Message(
                content=body,
                sender=from_number,
                channel="sms",
                message_id=message_sid if message_sid else f"sms_{from_number}_{int(time.time()*1000)}"
            )
            parent_self._handle_input_message(message)
            
            return {
                'status_code': 200,
                'headers': {'Content-Type': 'text/plain'},
                'body': ''
            }
        
        def handle_whatsapp_webhook(data: dict) -> dict:
            """Handle WhatsApp webhook requests"""
            from_number = data.get('From', '')
            body = data.get('Body', '')
            message_sid = data.get('MessageSid', '')  # Twilio's unique message ID
            
            log_message("INFO", f"[WEBHOOK] Parsed WhatsApp message from {from_number}: {body}")
            
            # Create and handle the message
            message = Message(
                content=body,
                sender=from_number,
                channel="whatsapp",
                message_id=message_sid if message_sid else f"whatsapp_{from_number}_{int(time.time()*1000)}"
            )
            parent_self._handle_input_message(message)
            
            return {
                'status_code': 200,
                'headers': {'Content-Type': 'text/plain'},
                'body': ''
            }
        
        # Register the endpoints
        api_server.register_endpoint('/sms', handle_sms_webhook)
        api_server.register_endpoint('/whatsapp', handle_whatsapp_webhook)
        
        log_message("INFO", f"Webhook endpoints registered with API server on port {api_server.config.port}")
        
        # Update webhook port to match API server
        self.config.webhook_port = api_server.config.port
        
        # Set up zrok if configured
        if self.config.webhook_use_tunnel:
            self._setup_zrok()
    
    def _extract_zrok_url(self, raw_url: str) -> str:
        """Extract clean URL from zrok output, removing markers like ││[PUBLIC]."""
        # Remove trailing period if any
        url = raw_url.rstrip('.')
        # Extract just the URL part before any ││ or other markers
        if '││' in url:
            url = url.split('││')[0]
        return url
    
    def _build_zrok_command(self) -> List[str]:
        """Build the zrok command based on configuration."""
        if self.config.zrok_reserved_token:
            # Use reserved share: zrok share reserved TOKEN --headless
            cmd = ["zrok", "share", "reserved", self.config.zrok_reserved_token, "--headless"]
            log_message("INFO", f"Using reserved share token: {self.config.zrok_reserved_token}")
        else:
            # Use ephemeral share: zrok share public http://localhost:PORT --headless
            cmd = ["zrok", "share", "public", f"http://localhost:{self.config.webhook_port}", "--headless"]
            log_message("INFO", "Using ephemeral zrok share")
        return cmd
    
    def _print_webhook_info(self):
        """Print webhook configuration information."""
        # Check if running in Claude wrapper mode
        shared_state = get_shared_state()
        is_wrapper_mode = shared_state.slack_mode_active or shared_state.whatsapp_mode_active
        
        has_sms = any(isinstance(p, TwilioSMSProvider) for p in self.providers)
        has_whatsapp = any(isinstance(p, TwilioWhatsAppProvider) for p in self.providers)
        
        # Log webhook info
        webhook_type = "Permanent" if self.config.zrok_reserved_token else "Temporary"
        log_message("INFO", f"{webhook_type} Webhook URLs:")
        if has_sms:
            log_message("INFO", f"  SMS: {self.webhook_url}/sms")
        if has_whatsapp:
            log_message("INFO", f"  WhatsApp: {self.webhook_url}/whatsapp")
            
        # Only print to console if not in wrapper mode
        if not is_wrapper_mode:
            if self.config.zrok_reserved_token:
                print("\nPermanent Webhook URLs:")
            else:
                print("\nTemporary Webhook URLs:")
            if has_sms:
                print(f"  SMS: {self.webhook_url}/sms")
            if has_whatsapp:
                print(f"  WhatsApp: {self.webhook_url}/whatsapp")
            
            if has_sms or has_whatsapp:
                print("\nAdd these URLs in your Twilio console:\n")
                print("  1. Go to https://console.twilio.com/")
                print("  2. For SMS: Phone Numbers → Manage → Active Numbers → Your Number → Messaging")
                print("  3. For WhatsApp: Messaging → Settings → WhatsApp Sandbox Settings")
                print("  4. Set the webhook URL for incoming messages")
            
            if not self.config.zrok_reserved_token:
                print("Note: This URL will change on restart. Use 'zrok reserve' for a permanent URL and set the")
                print("ZROK_RESERVED_TOKEN env variable so you won't need to update Twilio settings on restart!")
    
    def _read_zrok_output(self, stream, stream_name: str, url_found: threading.Event):
        """Read and parse zrok process output."""
        try:
            for line in iter(stream.readline, ''):
                if line.strip():  # Only log non-empty lines
                    log_message("DEBUG", f"Zrok {stream_name}: {line.strip()}")
                
                # Check for URL in the output
                if ".share.zrok.io" in line or "zrok.io" in line:
                    # Try to find URL in the line - stop at quotes, commas, or whitespace
                    match = re.search(r'https?://[^\s",}]+', line)
                    if match:
                        self.webhook_url = self._extract_zrok_url(match.group(0))
                        log_message("INFO", f"Zrok tunnel created: {self.webhook_url}")
                        url_found.set()
                        return
                
                # Sometimes the URL appears on its own line
                url_match = re.search(r'https?://[^\s",}]*\.zrok\.io', line)
                if url_match:
                    self.webhook_url = self._extract_zrok_url(url_match.group(0))
                    log_message("INFO", f"Zrok tunnel created: {self.webhook_url}")
                    url_found.set()
                    return
                
                # Check for error messages
                if "error" in line.lower() or "failed" in line.lower():
                    log_message("ERROR", f"Zrok error: {line}")
                
                # Check for specific errors
                if "subdomain is already registered" in line.lower():
                    log_message("ERROR", f"Zrok error: {line}")
        except Exception as e:
            log_message("ERROR", f"Error reading zrok {stream_name}: {e}")
    
    def _setup_zrok(self):
        """Set up zrok (free, open source, zero trust tunneling)"""
        try:
            # Build zrok command
            zrok_cmd = self._build_zrok_command()
            
            log_message("INFO", "Starting zrok tunnel...")
            
            # Start zrok process
            # Create a new process group so zrok doesn't interfere with terminal
            self.tunnel_process = subprocess.Popen(
                zrok_cmd,
                stdin=subprocess.DEVNULL,  # Prevent zrok from capturing stdin
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=0,  # Unbuffered for immediate output
                preexec_fn=os.setsid if os.name != 'nt' else None  # New process group
            )
            
            # Flag to track if URL was found
            url_found = threading.Event()
            
            # Create output reader function
            def read_output(stream, stream_name):
                self._read_zrok_output(stream, stream_name, url_found)
            
            # Start reading both stdout and stderr in background
            stdout_thread = threading.Thread(target=read_output, args=(self.tunnel_process.stdout, "stdout"), daemon=True)
            stderr_thread = threading.Thread(target=read_output, args=(self.tunnel_process.stderr, "stderr"), daemon=True)
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for URL with timeout
            if url_found.wait(timeout=20):
                self._print_webhook_info()
            else:
                log_message("WARNING", "Zrok started but URL not detected yet")
                print("\n⚠️  Zrok is starting...")
                print("Check if zrok is installed: https://github.com/openziti/zrok/releases")
                
        except FileNotFoundError:
            log_message("ERROR", "zrok command not found")
            print("\n❌ Zrok command not found. Please install it:")
            print("   Download from: https://github.com/openziti/zrok/releases/latest")
            print("   Or follow instructions at: https://docs.zrok.io/docs/getting-started/")
        except Exception as e:
            log_message("ERROR", f"Failed to create zrok tunnel: {e}")
            print(f"\n❌ Failed to create zrok tunnel: {e}")
            print("   Make sure zrok is installed and configured:")
            print("   1. Download: https://github.com/openziti/zrok/releases/latest")
            print("   2. Run: zrok enable <token> (if using reserved shares)")


# Convenience functions for integration
def create_config_from_env() -> CommsConfig:
    """Create configuration from environment variables"""
    config = CommsConfig()
    
    # Parse recipient lists from env
    sms_recipients = os.environ.get('SMS_RECIPIENTS', '')
    if sms_recipients:
        config.sms_recipients = [r.strip() for r in sms_recipients.split(',')]
    
    whatsapp_recipients = os.environ.get('WHATSAPP_RECIPIENTS', '')
    if whatsapp_recipients:
        config.whatsapp_recipients = [r.strip() for r in whatsapp_recipients.split(',')]
    
    return config


def setup_communication(providers: List[str] = None, config: Optional[CommsConfig] = None) -> Optional[CommunicationManager]:
    """Set up communication manager with specified providers and config"""
    log_message("INFO", f"Setting up communication with providers: {providers}")
    
    if config is None:
        config = create_config_from_env()
    
    # Log configuration details for debugging
    log_message("DEBUG", f"Config loaded - twilio_account_sid: {'***' if config.twilio_account_sid else 'None'}")
    log_message("DEBUG", f"Config loaded - twilio_whatsapp_number: {config.twilio_whatsapp_number}")
    log_message("DEBUG", f"Config loaded - slack_bot_token: {'***' if config.slack_bot_token else 'None'}")
    log_message("DEBUG", f"Config loaded - slack_app_token: {'***' if config.slack_app_token else 'None'}")
    log_message("DEBUG", f"Config loaded - slack_channel: {config.slack_channel}")
    log_message("DEBUG", f"ZROK_RESERVED_TOKEN set: {bool(os.environ.get('ZROK_RESERVED_TOKEN'))}")
    
    if not providers:
        # Auto-detect based on available credentials
        providers = []
        if config.twilio_account_sid and config.sms_recipients:
            providers.append("sms")
        if config.twilio_whatsapp_number and config.whatsapp_recipients:
            providers.append("whatsapp")
        if config.slack_bot_token and config.slack_app_token and config.slack_channel:
            providers.append("slack")
    
    log_message("INFO", f"Detected/configured providers: {providers}")
    
    if not providers:
        log_message("WARNING", "No communication providers configured or detected")
        return None
    
    manager = CommunicationManager(config)
    successfully_added = []
    for provider in providers:
        log_message("INFO", f"Adding provider: {provider}")
        if manager.add_provider(provider):
            successfully_added.append(provider)
        else:
            log_message("ERROR", f"Failed to add provider: {provider}")
    
    if not successfully_added:
        log_message("ERROR", "No providers could be added")
        return None
    
    manager.start()
    log_message("INFO", f"Communication manager started with providers: {', '.join(successfully_added)}")
    
    return manager


def run_interactive_mode(manager: CommunicationManager, providers: List[str]):
    """Run interactive messaging mode."""
    import termios

    providers_str = ', '.join(p.upper() for p in providers)
    print(f"\n{providers_str} Communication Active")
    if not manager.webhook_url:
        print(f"Webhook server listening on port {manager.config.webhook_port}")

    print("Waiting for incoming messages... (Ctrl+C to exit)")
    print("Type messages to send, press Enter to send.")

    # Wait for zrok to be ready
    print("\nWaiting for tunnel to be ready...")
    time.sleep(3)
    
    # Clear any buffered input and reset terminal
    try:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass
        
    # Now start interactive mode
    print("\nReady for interactive messaging. Type your message and press Enter.")
    print("> ", end='', flush=True)
    
    # Check if we can read from stdin
    active = True
    
    def handle_incoming():
        """Handle incoming messages in background"""
        while active:
            try:
                incoming = manager.get_input(timeout=0.5)
                if incoming:
                    print(f"\n[RECEIVED] {incoming}")
                    print("> ", end='', flush=True)
            except Exception:
                pass
    
    # Start incoming message handler
    incoming_thread = threading.Thread(target=handle_incoming, daemon=True)
    incoming_thread.start()
    
    # Main thread handles stdin
    try:
        while True:
            try:
                # Direct stdin read - no threading for input
                line = input()
                if line:
                    print(f"[SENDING] {line}")
                    manager.send_output(line)
                    print("> ", end='', flush=True)
            except EOFError:
                break
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        active = False


if __name__ == "__main__":
    # Make module runnable: python -m talkito.comms sms --sms-recipients +1234567890 "hello"
    import argparse

    parser = argparse.ArgumentParser(
        description="Send messages via Talkito communication providers",
        usage='%(prog)s [options] [MESSAGE]'
    )
    parser.add_argument("message", nargs='?', default=None,
                        help="Message to send (if not provided, enters interactive mode)")
    
    # Provider-specific options
    parser.add_argument("--sms-recipients", 
                        help="SMS recipient phone numbers (comma-separated)")
    parser.add_argument("--whatsapp-recipients",
                        help="WhatsApp recipient phone numbers (comma-separated)")
    parser.add_argument("--slack-channel",
                        help="Slack channel to send to")
    parser.add_argument("--log-file", 
                        help="Log file path (logging disabled if not specified)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    parser.add_argument("--webhook-port", type=int, default=8080,
                        help="Port for webhook server (default: 8080)")
    parser.add_argument("--zrok-reserved-token", 
                        help="Zrok reserved share token (from 'zrok reserve public')")
    parser.add_argument("--no-tunnel", action="store_true",
                        help="Disable zrok tunnel for webhooks")
    
    args = parser.parse_args()
    
    # Set up logging
    if args.log_file:
        from .logs import setup_logging
        setup_logging(args.log_file)
        print(f"Logging to: {args.log_file}")
        
        # Temporarily set DEBUG for zrok troubleshooting
        if args.log_level != "DEBUG":
            log_message("INFO", "Setting log level to DEBUG for zrok troubleshooting")
        
        # Suppress only the most verbose library outputs
        import logging as stdlib_logging
        stdlib_logging.getLogger('twilio.http_client').setLevel(stdlib_logging.WARNING)
    
    # Create config from environment
    config = create_config_from_env()
    
    # Override with command line arguments
    if args.sms_recipients:
        config.sms_recipients = [r.strip() for r in args.sms_recipients.split(',')]
    if args.whatsapp_recipients:
        config.whatsapp_recipients = [r.strip() for r in args.whatsapp_recipients.split(',')]
    if args.slack_channel:
        config.slack_channel = args.slack_channel
    if args.webhook_port:
        config.webhook_port = args.webhook_port
    if args.zrok_reserved_token:
        config.zrok_reserved_token = args.zrok_reserved_token
    if args.no_tunnel:
        config.webhook_use_tunnel = False
    
    # Auto-detect providers based on recipients
    providers = []
    if config.sms_recipients:
        providers.append("sms")
    if config.whatsapp_recipients:
        providers.append("whatsapp")
    if config.slack_channel and config.slack_bot_token and config.slack_app_token:
        providers.append("slack")
    
    if not providers:
        print("Error: No recipients configured. Use one or more of:")
        print("  --sms-recipients or set SMS_RECIPIENTS")
        print("  --whatsapp-recipients or set WHATSAPP_RECIPIENTS")
        print("  --slack-channel or set SLACK_CHANNEL")
        sys.exit(1)
    
    # Set up communication
    try:
        manager = setup_communication(providers, config)
        if not manager:
            print(f"Error: Failed to set up providers: {', '.join(providers)}. Check credentials.")
            sys.exit(1)
        
        # Get actual active providers
        active_providers = manager.get_active_provider_types()
        
        # If message provided, send it
        if args.message:
            providers_str = ', '.join(active_providers)
            print(f"Sending message via {providers_str}...")
            manager.send_output(args.message)
            time.sleep(2)
            print("Message sent!")
        
        # Enter interactive mode
        run_interactive_mode(manager, active_providers)
        
        manager.stop()
        print("\nShutting down...")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
