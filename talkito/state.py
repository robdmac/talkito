"""
Shared state manager for Talkito MCP and wrapper integration.
This ensures configuration changes made through MCP tools affect the wrapper runtime.
"""

import threading
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
import json


@dataclass
class CommunicationConfig:
    """Communication channel configuration"""
    # WhatsApp
    whatsapp_enabled: bool = False
    whatsapp_from_number: Optional[str] = None
    whatsapp_to_number: Optional[str] = None
    whatsapp_account_sid: Optional[str] = None
    whatsapp_auth_token: Optional[str] = None
    
    # Slack
    slack_enabled: bool = False
    slack_channel: Optional[str] = None
    slack_bot_token: Optional[str] = None
    slack_app_token: Optional[str] = None


@dataclass 
class TalkitoState:
    """Shared state for all Talkito components"""
    # Core features
    tts_enabled: bool = True
    asr_enabled: bool = True
    tts_initialized: bool = False
    asr_initialized: bool = False
    
    # Provider preferences
    tts_provider: Optional[str] = None
    asr_provider: Optional[str] = None
    
    # TTS configuration details
    tts_voice: Optional[str] = None
    tts_region: Optional[str] = None
    tts_language: Optional[str] = None
    tts_rate: Optional[float] = None
    tts_pitch: Optional[float] = None
    
    # ASR configuration details
    asr_language: Optional[str] = None
    asr_model: Optional[str] = None
    
    # Communication modes
    whatsapp_mode_active: bool = False
    slack_mode_active: bool = False
    
    # Communication config
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    
    # Active modes
    voice_mode_active: bool = False
    
    # Callbacks for state changes
    _callbacks: Dict[str, List[Callable]] = field(default_factory=dict)
    
    # Thread safety lock
    _lock: threading.RLock = field(default_factory=threading.RLock)
    
    def register_callback(self, event: str, callback: Callable):
        """Register a callback for state change events"""
        with self._lock:
            if event not in self._callbacks:
                self._callbacks[event] = []
            self._callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, **kwargs):
        """Trigger all callbacks for an event"""
        # Get callbacks while holding lock
        with self._lock:
            callbacks = self._callbacks.get(event, []).copy()
        
        # Execute callbacks outside lock to avoid deadlock
        for callback in callbacks:
            try:
                callback(**kwargs)
            except Exception as e:
                print(f"Error in callback for {event}: {e}")
    
    def set_tts_enabled(self, enabled: bool):
        """Set TTS enabled state and trigger callbacks"""
        with self._lock:
            old_value = self.tts_enabled
            self.tts_enabled = enabled
            changed = old_value != enabled
        
        if changed:
            self._trigger_callbacks('tts_changed', enabled=enabled)
    
    def set_asr_enabled(self, enabled: bool):
        """Set ASR enabled state and trigger callbacks"""
        with self._lock:
            old_value = self.asr_enabled
            self.asr_enabled = enabled
            changed = old_value != enabled
        
        if changed:
            self._trigger_callbacks('asr_changed', enabled=enabled)
    
    def set_tts_initialized(self, initialized: bool, provider: Optional[str] = None):
        """Set TTS initialization state"""
        with self._lock:
            self.tts_initialized = initialized
            if provider:
                self.tts_provider = provider
        self._trigger_callbacks('tts_initialized', initialized=initialized, provider=provider)
    
    def set_asr_initialized(self, initialized: bool, provider: Optional[str] = None):
        """Set ASR initialization state"""
        with self._lock:
            self.asr_initialized = initialized
            if provider:
                self.asr_provider = provider
        self._trigger_callbacks('asr_initialized', initialized=initialized, provider=provider)
    
    def set_voice_mode(self, active: bool):
        """Set voice mode (both TTS and ASR)"""
        with self._lock:
            self.voice_mode_active = active
        self.set_tts_enabled(active)
        self.set_asr_enabled(active)
        self._trigger_callbacks('voice_mode_changed', active=active)
    
    def set_whatsapp_mode(self, active: bool):
        """Set WhatsApp mode state"""
        with self._lock:
            old_value = self.whatsapp_mode_active
            self.whatsapp_mode_active = active
            changed = old_value != active
        
        if changed:
            self._trigger_callbacks('whatsapp_mode_changed', active=active)
    
    def set_slack_mode(self, active: bool):
        """Set Slack mode state"""
        with self._lock:
            old_value = self.slack_mode_active
            self.slack_mode_active = active
            changed = old_value != active
        
        if changed:
            self._trigger_callbacks('slack_mode_changed', active=active)
    
    def set_tts_config(self, provider: Optional[str] = None, voice: Optional[str] = None, 
                       region: Optional[str] = None, language: Optional[str] = None,
                       rate: Optional[float] = None, pitch: Optional[float] = None):
        """Set TTS configuration"""
        with self._lock:
            if provider is not None:
                self.tts_provider = provider
            if voice is not None:
                self.tts_voice = voice
            if region is not None:
                self.tts_region = region
            if language is not None:
                self.tts_language = language
            if rate is not None:
                self.tts_rate = rate
            if pitch is not None:
                self.tts_pitch = pitch
        self._trigger_callbacks('tts_config_changed', provider=provider, voice=voice)
    
    def set_asr_config(self, provider: Optional[str] = None, language: Optional[str] = None,
                       model: Optional[str] = None):
        """Set ASR configuration"""
        with self._lock:
            if provider is not None:
                self.asr_provider = provider
            if language is not None:
                self.asr_language = language
            if model is not None:
                self.asr_model = model
        self._trigger_callbacks('asr_config_changed', provider=provider, language=language, model=model)
    
    def turn_off_all(self):
        """Master off switch - disable all modes"""
        self.set_voice_mode(False)
        self.set_whatsapp_mode(False)
        self.set_slack_mode(False)
        self._trigger_callbacks('all_modes_off')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization"""
        with self._lock:
            return {
                'tts_enabled': self.tts_enabled,
                'asr_enabled': self.asr_enabled,
                'tts_initialized': self.tts_initialized,
                'asr_initialized': self.asr_initialized,
                'tts_provider': self.tts_provider,
                'tts_voice': self.tts_voice,
                'tts_region': self.tts_region,
                'tts_language': self.tts_language,
                'tts_rate': self.tts_rate,
                'tts_pitch': self.tts_pitch,
                'asr_provider': self.asr_provider,
                'asr_language': self.asr_language,
                'asr_model': self.asr_model,
                'whatsapp_mode_active': self.whatsapp_mode_active,
                'slack_mode_active': self.slack_mode_active,
                'voice_mode_active': self.voice_mode_active,
                'communication': {
                    'whatsapp_enabled': self.communication.whatsapp_enabled,
                    'whatsapp_from_number': self.communication.whatsapp_from_number,
                    'whatsapp_to_number': self.communication.whatsapp_to_number,
                    'slack_enabled': self.communication.slack_enabled,
                    'slack_channel': self.communication.slack_channel,
                }
            }
    
    # Thread-safe property getters
    def get_tts_enabled(self) -> bool:
        """Thread-safe getter for tts_enabled"""
        with self._lock:
            return self.tts_enabled
    
    def get_asr_enabled(self) -> bool:
        """Thread-safe getter for asr_enabled"""
        with self._lock:
            return self.asr_enabled
    
    def get_tts_initialized(self) -> bool:
        """Thread-safe getter for tts_initialized"""
        with self._lock:
            return self.tts_initialized
    
    def get_asr_initialized(self) -> bool:
        """Thread-safe getter for asr_initialized"""
        with self._lock:
            return self.asr_initialized
    
    def get_whatsapp_mode_active(self) -> bool:
        """Thread-safe getter for whatsapp_mode_active"""
        with self._lock:
            return self.whatsapp_mode_active
    
    def get_slack_mode_active(self) -> bool:
        """Thread-safe getter for slack_mode_active"""
        with self._lock:
            return self.slack_mode_active
    
    def get_tts_provider(self) -> Optional[str]:
        """Thread-safe getter for tts_provider"""
        with self._lock:
            return self.tts_provider
    
    def get_asr_provider(self) -> Optional[str]:
        """Thread-safe getter for asr_provider"""
        with self._lock:
            return self.asr_provider


class SharedStateManager:
    """Singleton manager for shared Talkito state"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.state = TalkitoState()
        self._initialized = True
        
        # Load saved state if available
        self._load_state()
    
    def _load_state(self):
        """Load persisted state from file if available"""
        try:
            import os
            state_file = os.path.expanduser('~/.talkito_state.json')
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    # Only load non-runtime state
                    if 'tts_provider' in data:
                        self.state.tts_provider = data['tts_provider']
                    if 'tts_voice' in data:
                        self.state.tts_voice = data['tts_voice']
                    if 'tts_region' in data:
                        self.state.tts_region = data['tts_region']
                    if 'tts_language' in data:
                        self.state.tts_language = data['tts_language']
                    if 'tts_rate' in data:
                        self.state.tts_rate = data['tts_rate']
                    if 'tts_pitch' in data:
                        self.state.tts_pitch = data['tts_pitch']
                    if 'asr_provider' in data:
                        self.state.asr_provider = data['asr_provider']
                    if 'asr_language' in data:
                        self.state.asr_language = data['asr_language']
                    if 'asr_model' in data:
                        self.state.asr_model = data['asr_model']
                    # Communication config
                    if 'communication' in data:
                        comm = data['communication']
                        if 'whatsapp_from_number' in comm:
                            self.state.communication.whatsapp_from_number = comm['whatsapp_from_number']
                        if 'whatsapp_to_number' in comm:
                            self.state.communication.whatsapp_to_number = comm['whatsapp_to_number']
                        if 'slack_channel' in comm:
                            self.state.communication.slack_channel = comm['slack_channel']
        except Exception:
            # Ignore errors loading state
            pass
    
    def save_state(self):
        """Persist state to file"""
        try:
            import os
            state_file = os.path.expanduser('~/.talkito_state.json')
            with open(state_file, 'w') as f:
                json.dump(self.state.to_dict(), f, indent=2)
        except Exception:
            # Ignore errors saving state
            pass
    
    def get_state(self) -> TalkitoState:
        """Get the shared state instance"""
        return self.state
    
    def reload_state(self):
        """Reload runtime state from file"""
        try:
            import os
            state_file = os.path.expanduser('~/.talkito_state.json')
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    # Reload runtime state flags
                    if 'tts_enabled' in data:
                        self.state.tts_enabled = data['tts_enabled']
                    if 'asr_enabled' in data:
                        self.state.asr_enabled = data['asr_enabled']
                    if 'voice_mode_active' in data:
                        self.state.voice_mode_active = data['voice_mode_active']
                    if 'whatsapp_mode_active' in data:
                        self.state.whatsapp_mode_active = data['whatsapp_mode_active']
                    if 'slack_mode_active' in data:
                        self.state.slack_mode_active = data['slack_mode_active']
        except Exception:
            # Ignore errors reloading state
            pass
    
    def reset(self):
        """Reset state to defaults (mainly for testing)"""
        with self._lock:
            self.state = TalkitoState()
    
    # Thread-safe state mutation methods
    def set_tts_config(self, provider: Optional[str] = None, voice: Optional[str] = None,
                       region: Optional[str] = None, language: Optional[str] = None,
                       rate: Optional[float] = None, pitch: Optional[float] = None):
        """Thread-safe TTS configuration update"""
        with self._lock:
            self.state.set_tts_config(provider, voice, region, language, rate, pitch)
            self.save_state()
    
    def set_tts_enabled(self, enabled: bool):
        """Thread-safe TTS enabled state update"""
        with self._lock:
            self.state.set_tts_enabled(enabled)
            self.save_state()
    
    def set_asr_enabled(self, enabled: bool):
        """Thread-safe ASR enabled state update"""
        with self._lock:
            self.state.set_asr_enabled(enabled)
            self.save_state()
    
    def set_asr_config(self, provider: Optional[str] = None, language: Optional[str] = None,
                       model: Optional[str] = None):
        """Thread-safe ASR configuration update"""
        with self._lock:
            self.state.set_asr_config(provider, language, model)
            self.save_state()
    
    def set_asr_initialized(self, initialized: bool, provider: Optional[str] = None):
        """Thread-safe ASR initialized state update"""
        with self._lock:
            self.state.set_asr_initialized(initialized, provider)
            self.save_state()
    
    def set_voice_mode(self, active: bool):
        """Thread-safe voice mode update"""
        with self._lock:
            self.state.set_voice_mode(active)
            self.save_state()
    
    def set_whatsapp_mode(self, active: bool):
        """Thread-safe WhatsApp mode update"""
        with self._lock:
            self.state.set_whatsapp_mode(active)
            self.save_state()
    
    def set_slack_mode(self, active: bool):
        """Thread-safe Slack mode update"""
        with self._lock:
            self.state.set_slack_mode(active)
            self.save_state()


# Global instance for easy access
_shared_state = SharedStateManager()


def get_shared_state() -> TalkitoState:
    """Get the global shared state instance"""
    return _shared_state.get_state()


def save_shared_state():
    """Save the current state to disk"""
    _shared_state.save_state()


def get_status_summary(comms_manager=None, whatsapp_recipient=None, slack_channel=None,
                       tts_override=False, asr_override=False, 
                       configured_tts_provider=None, configured_asr_provider=None) -> str:
    """Generate a one-line status summary for TalkiTo components
    
    Args:
        comms_manager: Communication manager instance (optional)
        whatsapp_recipient: Current WhatsApp recipient (optional)
        slack_channel: Current Slack channel (optional)
        tts_override: Whether TTS is being overridden to show as enabled
        asr_override: Whether ASR is being overridden to show as enabled
        configured_tts_provider: TTS provider from command line args (optional)
        configured_asr_provider: ASR provider from command line args (optional)
    
    Returns:
        One-line formatted status string
    """
    try:
        # Import needed modules
        from . import tts
        try:
            from . import asr
            ASR_AVAILABLE = True
        except ImportError:
            asr = None
            ASR_AVAILABLE = False
        
        # Import provider types if comms_manager is provided
        if comms_manager:
            from .comms import SlackProvider, TwilioWhatsAppProvider
        
        shared_state = get_shared_state()
        
        # Build status dict - use thread-safe methods
        status = {
            "tts": {
                "initialized": shared_state.get_tts_initialized(),
                "enabled": shared_state.get_tts_enabled(),
                "is_speaking": tts.is_speaking() if shared_state.get_tts_initialized() else False,
                "provider": configured_tts_provider or shared_state.get_tts_provider() or tts.select_best_tts_provider()
            },
            "asr": {
                "available": ASR_AVAILABLE,
                "initialized": shared_state.get_asr_initialized(),
                "enabled": shared_state.get_asr_enabled(),
                "is_listening": asr.is_dictation_active() if ASR_AVAILABLE and shared_state.get_asr_initialized() else False,
                "provider": configured_asr_provider or shared_state.get_asr_provider() or (asr.select_best_asr_provider() if ASR_AVAILABLE else None)
            },
            "whatsapp": {
                "mode_active": shared_state.get_whatsapp_mode_active(),
                "recipient": whatsapp_recipient,
                "configured": comms_manager is not None and any(isinstance(p, TwilioWhatsAppProvider) for p in (comms_manager.providers if comms_manager else []))
            },
            "slack": {
                "mode_active": shared_state.get_slack_mode_active(),
                "channel": slack_channel,
                "configured": comms_manager is not None and any(isinstance(p, SlackProvider) for p in (comms_manager.providers if comms_manager else []))
            }
        }
        
        # Format as a one-line summary
        # Show green only if both initialized AND enabled
        if tts_override:
            tts_emoji = "🟢"
        else:
            tts_emoji = "🟢" if status["tts"]["initialized"] and status["tts"]["enabled"] else "🔴"
        if asr_override:
            asr_emoji = "🟢"
        else:
            asr_emoji = "🟢" if status["asr"]["initialized"] and status["asr"]["enabled"] else "🔴"
        
        # Communication status - check shared state instead of comm_manager
        comms = []
        if shared_state.get_whatsapp_mode_active():
            comms.append("🟢 WhatsApp")
        if shared_state.get_slack_mode_active():
            comms.append("🟢 Slack")
        
        return f"TalkiTo: {tts_emoji} TTS ({status['tts']['provider']}) | {asr_emoji} ASR ({status['asr']['provider'] or 'none'}) | Comms: {', '.join(comms) if comms else 'none'}"
        
    except Exception as e:
        return f"Error getting status: {str(e)}"


def set_tts_config_thread_safe(provider: Optional[str] = None, voice: Optional[str] = None,
                               region: Optional[str] = None, language: Optional[str] = None,
                               rate: Optional[float] = None, pitch: Optional[float] = None):
    """Thread-safe helper to update TTS configuration"""
    _shared_state.set_tts_config(provider, voice, region, language, rate, pitch)


def set_asr_config_thread_safe(provider: Optional[str] = None, language: Optional[str] = None,
                               model: Optional[str] = None):
    """Thread-safe helper to update ASR configuration"""
    _shared_state.set_asr_config(provider, language, model)