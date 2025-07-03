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
config.py - Centralized configuration management for talkito
Single source of truth for all configuration values
"""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

# Try to load .env files if available
try:
    from dotenv import load_dotenv
    # Load .env first (takes precedence)
    load_dotenv()
    # Also load .talkito.env (won't override existing vars from .env)
    load_dotenv('.talkito.env')
except ImportError:
    pass


@dataclass
class TalkitoConfig:
    """Centralized configuration for all talkito components"""
    
    # Core settings
    log_file: Optional[str] = None
    verbosity: int = 0
    profile: str = "default"
    
    # TTS settings
    tts_enabled: bool = True
    tts_provider: Optional[str] = None
    tts_voice: Optional[str] = None
    tts_region: Optional[str] = None
    tts_language: Optional[str] = None
    tts_rate: Optional[float] = None
    tts_pitch: Optional[float] = None
    
    # ASR settings
    asr_enabled: bool = True
    asr_provider: Optional[str] = None
    asr_language: str = "en-US"
    asr_auto_listen: bool = True
    asr_tap_to_talk: bool = False
    asr_min_silence_ms: int = 1500
    
    # Communication settings
    comms_webhook_port: int = 3000
    comms_webhook_timeout: int = 20
    
    # Twilio settings
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_phone_number: Optional[str] = None
    twilio_whatsapp_number: Optional[str] = None
    
    # Slack settings
    slack_bot_token: Optional[str] = None
    slack_app_token: Optional[str] = None
    slack_channel: Optional[str] = None
    
    # Recipient lists
    sms_recipients: List[str] = field(default_factory=list)
    whatsapp_recipients: List[str] = field(default_factory=list)
    
    # Provider API keys
    openai_api_key: Optional[str] = None
    deepgram_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None
    azure_speech_key: Optional[str] = None
    azure_speech_region: Optional[str] = None
    
    # AWS credentials (for Polly)
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: Optional[str] = None
    
    # Google Cloud credentials
    google_application_credentials: Optional[str] = None
    gcloud_service_account_json: Optional[str] = None
    
    # Advanced settings
    max_speech_length: int = 500
    similarity_threshold: float = 0.85
    recent_text_cache_size: int = 10
    debounce_time: float = 1.0
    
    # Feature flags
    auto_skip_tts: bool = False
    clean_text_flag: bool = True
    
    @classmethod
    def from_env(cls) -> 'TalkitoConfig':
        """Create configuration from environment variables"""
        config = cls()
        
        # Core settings
        config.log_file = os.environ.get('TALKITO_LOG_FILE')
        config.verbosity = int(os.environ.get('TALKITO_VERBOSITY', '0'))
        config.profile = os.environ.get('TALKITO_PROFILE', 'default')
        
        # TTS settings
        config.tts_enabled = os.environ.get('TALKITO_TTS_ENABLED', 'true').lower() == 'true'
        config.tts_provider = os.environ.get('TALKITO_PREFERRED_TTS_PROVIDER')
        config.tts_voice = os.environ.get('TALKITO_TTS_VOICE')
        config.tts_region = os.environ.get('TALKITO_TTS_REGION')
        config.tts_language = os.environ.get('TALKITO_TTS_LANGUAGE')
        
        rate = os.environ.get('TALKITO_TTS_RATE')
        if rate:
            config.tts_rate = float(rate)
            
        pitch = os.environ.get('TALKITO_TTS_PITCH')
        if pitch:
            config.tts_pitch = float(pitch)
        
        # ASR settings
        config.asr_enabled = os.environ.get('TALKITO_ASR_ENABLED', 'true').lower() == 'true'
        config.asr_provider = os.environ.get('TALKITO_PREFERRED_ASR_PROVIDER')
        config.asr_language = os.environ.get('TALKITO_ASR_LANGUAGE', 'en-US')
        config.asr_auto_listen = os.environ.get('TALKITO_ASR_AUTO_LISTEN', 'true').lower() == 'true'
        config.asr_tap_to_talk = os.environ.get('TALKITO_ASR_TAP_TO_TALK', 'false').lower() == 'true'
        config.asr_min_silence_ms = int(os.environ.get('TALKITO_ASR_MIN_SILENCE_MS', '1500'))
        
        # Communication settings
        config.comms_webhook_port = int(os.environ.get('WEBHOOK_PORT', '3000'))
        config.comms_webhook_timeout = int(os.environ.get('WEBHOOK_TIMEOUT', '20'))
        
        # Twilio settings
        config.twilio_account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
        config.twilio_auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
        config.twilio_phone_number = os.environ.get('TWILIO_PHONE_NUMBER')
        config.twilio_whatsapp_number = os.environ.get('TWILIO_WHATSAPP_NUMBER')
        
        # Slack settings
        config.slack_bot_token = os.environ.get('SLACK_BOT_TOKEN')
        config.slack_app_token = os.environ.get('SLACK_APP_TOKEN')
        config.slack_channel = os.environ.get('SLACK_CHANNEL')
        
        # Recipient lists
        sms_recipients = os.environ.get('SMS_RECIPIENTS', '')
        if sms_recipients:
            config.sms_recipients = [r.strip() for r in sms_recipients.split(',')]
            
        whatsapp_recipients = os.environ.get('WHATSAPP_RECIPIENTS', '')
        if whatsapp_recipients:
            config.whatsapp_recipients = [r.strip() for r in whatsapp_recipients.split(',')]
        
        # Provider API keys
        config.openai_api_key = os.environ.get('OPENAI_API_KEY')
        config.deepgram_api_key = os.environ.get('DEEPGRAM_API_KEY')
        config.elevenlabs_api_key = os.environ.get('ELEVENLABS_API_KEY')
        config.azure_speech_key = os.environ.get('AZURE_SPEECH_KEY')
        config.azure_speech_region = os.environ.get('AZURE_SPEECH_REGION')
        
        # AWS credentials
        config.aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
        config.aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        config.aws_region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
        
        # Google Cloud credentials
        config.google_application_credentials = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        config.gcloud_service_account_json = os.environ.get('GCLOUD_SERVICE_ACCOUNT_JSON')
        
        # Advanced settings
        config.max_speech_length = int(os.environ.get('TALKITO_MAX_SPEECH_LENGTH', '500'))
        config.similarity_threshold = float(os.environ.get('TALKITO_SIMILARITY_THRESHOLD', '0.85'))
        config.recent_text_cache_size = int(os.environ.get('TALKITO_RECENT_TEXT_CACHE_SIZE', '10'))
        config.debounce_time = float(os.environ.get('TALKITO_DEBOUNCE_TIME', '1.0'))
        
        # Feature flags
        config.auto_skip_tts = os.environ.get('TALKITO_AUTO_SKIP_TTS', 'false').lower() == 'true'
        config.clean_text_flag = os.environ.get('TALKITO_CLEAN_TEXT', 'true').lower() == 'true'
        
        return config
    
    def merge_with_args(self, args: Any) -> None:
        """Merge command-line arguments with configuration"""
        if hasattr(args, 'log_file') and args.log_file:
            self.log_file = args.log_file
            
        if hasattr(args, 'verbose'):
            # Count verbose flags (-v = 1, -vv = 2, etc.)
            self.verbosity = args.verbose or 0
            
        if hasattr(args, 'profile') and args.profile:
            self.profile = args.profile
            
        if hasattr(args, 'provider') and args.provider:
            self.tts_provider = args.provider
            
        if hasattr(args, 'voice') and args.voice:
            self.tts_voice = args.voice
            
        if hasattr(args, 'region') and args.region:
            self.tts_region = args.region
            
        if hasattr(args, 'language') and args.language:
            self.tts_language = args.language
            
        if hasattr(args, 'no_tts') and args.no_tts:
            self.tts_enabled = False
            
        if hasattr(args, 'no_asr') and args.no_asr:
            self.asr_enabled = False
            
        if hasattr(args, 'asr_provider') and args.asr_provider:
            self.asr_provider = args.asr_provider
            
        if hasattr(args, 'asr_language') and args.asr_language:
            self.asr_language = args.asr_language
            
        if hasattr(args, 'tap_to_talk') and args.tap_to_talk:
            self.asr_tap_to_talk = True
            self.asr_auto_listen = False
            
        if hasattr(args, 'auto_skip_tts') and args.auto_skip_tts:
            self.auto_skip_tts = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    def get_tts_config(self) -> Dict[str, Any]:
        """Get TTS-specific configuration"""
        return {
            'provider': self.tts_provider,
            'voice': self.tts_voice,
            'region': self.tts_region,
            'language': self.tts_language,
            'rate': self.tts_rate,
            'pitch': self.tts_pitch,
        }
    
    def get_asr_config(self) -> Dict[str, Any]:
        """Get ASR-specific configuration"""
        return {
            'provider': self.asr_provider,
            'language': self.asr_language,
            'auto_listen': self.asr_auto_listen,
            'tap_to_talk': self.asr_tap_to_talk,
            'min_silence_ms': self.asr_min_silence_ms,
        }
    
    def get_comms_config(self) -> Dict[str, Any]:
        """Get communication-specific configuration"""
        return {
            'webhook_port': self.comms_webhook_port,
            'webhook_timeout': self.comms_webhook_timeout,
            'twilio_account_sid': self.twilio_account_sid,
            'twilio_auth_token': self.twilio_auth_token,
            'twilio_phone_number': self.twilio_phone_number,
            'twilio_whatsapp_number': self.twilio_whatsapp_number,
            'slack_bot_token': self.slack_bot_token,
            'slack_app_token': self.slack_app_token,
            'slack_channel': self.slack_channel,
            'sms_recipients': self.sms_recipients,
            'whatsapp_recipients': self.whatsapp_recipients,
        }


# Global configuration instance
_config: Optional[TalkitoConfig] = None


def get_config() -> TalkitoConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = TalkitoConfig.from_env()
    return _config


def reload_config() -> TalkitoConfig:
    """Reload configuration from environment"""
    global _config
    _config = TalkitoConfig.from_env()
    return _config


def set_config(config: TalkitoConfig) -> None:
    """Set the global configuration instance"""
    global _config
    _config = config