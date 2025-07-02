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
    
    # Provider preferences
    tts_provider: Optional[str] = None
    asr_provider: Optional[str] = None
    
    # Communication modes
    whatsapp_mode_active: bool = False
    slack_mode_active: bool = False
    
    # Communication config
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    
    # Active modes
    voice_mode_active: bool = False
    
    # Callbacks for state changes
    _callbacks: Dict[str, List[Callable]] = field(default_factory=dict)
    
    def register_callback(self, event: str, callback: Callable):
        """Register a callback for state change events"""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, **kwargs):
        """Trigger all callbacks for an event"""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    callback(**kwargs)
                except Exception as e:
                    print(f"Error in callback for {event}: {e}")
    
    def set_tts_enabled(self, enabled: bool):
        """Set TTS enabled state and trigger callbacks"""
        old_value = self.tts_enabled
        self.tts_enabled = enabled
        if old_value != enabled:
            self._trigger_callbacks('tts_changed', enabled=enabled)
    
    def set_asr_enabled(self, enabled: bool):
        """Set ASR enabled state and trigger callbacks"""
        old_value = self.asr_enabled
        self.asr_enabled = enabled
        if old_value != enabled:
            self._trigger_callbacks('asr_changed', enabled=enabled)
    
    def set_voice_mode(self, active: bool):
        """Set voice mode (both TTS and ASR)"""
        self.voice_mode_active = active
        self.set_tts_enabled(active)
        self.set_asr_enabled(active)
        self._trigger_callbacks('voice_mode_changed', active=active)
    
    def set_whatsapp_mode(self, active: bool):
        """Set WhatsApp mode state"""
        old_value = self.whatsapp_mode_active
        self.whatsapp_mode_active = active
        if old_value != active:
            self._trigger_callbacks('whatsapp_mode_changed', active=active)
    
    def set_slack_mode(self, active: bool):
        """Set Slack mode state"""
        old_value = self.slack_mode_active
        self.slack_mode_active = active
        if old_value != active:
            self._trigger_callbacks('slack_mode_changed', active=active)
    
    def turn_off_all(self):
        """Master off switch - disable all modes"""
        self.set_voice_mode(False)
        self.set_whatsapp_mode(False)
        self.set_slack_mode(False)
        self._trigger_callbacks('all_modes_off')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization"""
        return {
            'tts_enabled': self.tts_enabled,
            'asr_enabled': self.asr_enabled,
            'tts_provider': self.tts_provider,
            'asr_provider': self.asr_provider,
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
                    if 'asr_provider' in data:
                        self.state.asr_provider = data['asr_provider']
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
    
    def reset(self):
        """Reset state to defaults (mainly for testing)"""
        self.state = TalkitoState()


# Global instance for easy access
_shared_state = SharedStateManager()


def get_shared_state() -> TalkitoState:
    """Get the global shared state instance"""
    return _shared_state.get_state()


def save_shared_state():
    """Save the current state to disk"""
    _shared_state.save_state()