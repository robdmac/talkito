"""Shared state manager for Talkito MCP and wrapper integration ensuring configuration changes made through MCP tools affect the wrapper runtime."""

import json
import os
import threading
import time
from pathlib import Path

from typing import Optional, Dict, Any, Callable, List, TYPE_CHECKING
from dataclasses import dataclass, field
from .logs import log_message

if TYPE_CHECKING:
    from .comms import CommsConfig


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_assignment(raw_line: str) -> Optional[tuple[str, str]]:
    line = raw_line.strip()
    if not line or line.startswith('#'):
        return None
    if line.startswith("export "):
        line = line[7:].strip()
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    if not key:
        return None
    return key, value.strip()


def _parse_key(line: str) -> Optional[str]:
    assignment = _parse_assignment(line)
    return assignment[0] if assignment else None


def load_dotenv(path: str = ".env", override: bool = False) -> bool:
    env_path = Path(path)
    if not env_path.exists():
        return False

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        assignment = _parse_assignment(raw_line)
        if not assignment:
            continue
        key, value = assignment
        value = _strip_quotes(value)
        if not override and key in os.environ:
            continue
        os.environ[key] = value
    return True


def set_key(path: str, key: str, value: Optional[str]) -> tuple[str, Optional[str], bool]:
    env_path = Path(path)
    lines = []
    found = False
    file_existed = env_path.exists()

    if file_existed:
        with env_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                parsed_key = _parse_key(line.strip())
                if parsed_key == key:
                    found = True
                    if value is not None:
                        lines.append(f"{key}={value}")
                else:
                    lines.append(line)

    if not found and value is not None:
        lines.append(f"{key}={value}")

    if not lines and not file_existed and value is None:
        return key, None, True

    env_path.parent.mkdir(parents=True, exist_ok=True)

    content = ("\n".join(lines) + "\n") if lines else ""
    with env_path.open("w", encoding="utf-8") as handle:
        handle.write(content)

    return key, value, True


def unset_key(path: str, key: str) -> tuple[str, None, bool]:
    return set_key(path, key, None)


# Load environment configuration on module import
load_dotenv()
load_dotenv('.talkito.env')


def _import_asr():
    from . import asr as _asr
    return _asr


def _resolve_comm_config_bool(config: "CommsConfig", attr_name: str, fallback_fields: List[str]) -> bool:
    """Resolve boolean flags for communication configuration.

    This prefers explicit *_enabled flags but falls back to checking supporting fields
    so we can infer configuration without fully initializing providers.
    """
    if config is None:
        return False

    explicit_flag = getattr(config, attr_name, None)
    if explicit_flag is not None:
        return bool(explicit_flag)

    return all(getattr(config, field, None) for field in fallback_fields)


def sync_communication_state_from_config(
    comms_config: Optional["CommsConfig"],
    *,
    slack_configured: Optional[bool] = None,
    whatsapp_configured: Optional[bool] = None,
) -> None:
    """Update shared communication state based on a CommsConfig.

    The optional *_configured overrides allow callers to record the actual provider
    status when available (e.g., after CommunicationManager initialization).
    """
    shared_state = get_shared_state()
    comm_state = shared_state.communication

    # Determine WhatsApp configuration
    has_whatsapp = whatsapp_configured
    if has_whatsapp is None:
        has_whatsapp = _resolve_comm_config_bool(
            comms_config,
            "whatsapp_enabled",
            ["twilio_whatsapp_number", "whatsapp_recipients"],
        )
    comm_state.whatsapp_enabled = bool(has_whatsapp)

    if has_whatsapp and comms_config:
        recipients = getattr(comms_config, "whatsapp_recipients", None) or []
        comm_state.whatsapp_to_number = recipients[0] if recipients else None
    else:
        comm_state.whatsapp_to_number = None

    # Determine Slack configuration
    has_slack = slack_configured
    if has_slack is None:
        has_slack = _resolve_comm_config_bool(
            comms_config,
            "slack_enabled",
            ["slack_bot_token", "slack_channel"],
        )
    comm_state.slack_enabled = bool(has_slack)

    if has_slack and comms_config:
        comm_state.slack_channel = getattr(comms_config, "slack_channel", None)
    else:
        comm_state.slack_channel = None

    log_message(
        "DEBUG",
        f"[STATE] sync_communication_state_from_config slack_enabled={comm_state.slack_enabled} "
        f"whatsapp_enabled={comm_state.whatsapp_enabled} "
        f"slack_channel={comm_state.slack_channel} "
        f"whatsapp_to={comm_state.whatsapp_to_number}",
    )


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
    tts_model: Optional[str] = None
    tts_rate: Optional[float] = None
    tts_pitch: Optional[float] = None
    tts_mode: str = 'auto-skip'
    
    # ASR configuration details
    asr_language: Optional[str] = None
    asr_model: Optional[str] = None
    asr_mode: str = 'auto-input'
    asr_source_file: Optional[str] = None  # File path for file: mode
    
    # Communication modes
    whatsapp_mode_active: bool = False
    slack_mode_active: bool = False
    
    # Communication config
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    
    # Active modes
    voice_mode_active: bool = False
    
    # Hook tracking
    in_tool_use: bool = False  # True between PreToolUse and PostToolUse hooks

    # MCP server tracking
    mcp_server_running: bool = False

    # One-time notifications
    tap_to_talk_notification_shown: bool = False
    
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
                       model: Optional[str] = None, rate: Optional[float] = None,
                       pitch: Optional[float] = None):
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
            if model is not None:
                self.tts_model = model
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
        """Convert state to dictionary for serialization - excluding runtime provider info"""
        with self._lock:
            return {
                'tts_enabled': self.tts_enabled,
                'asr_enabled': self.asr_enabled,
                'tts_initialized': self.tts_initialized,
                'asr_initialized': self.asr_initialized,
                # Persist provider info for background MCP processes
                'tts_provider': self.tts_provider,
                'asr_provider': self.asr_provider,
                'tts_voice': self.tts_voice,
                'tts_region': self.tts_region,
                'tts_language': self.tts_language,
                'tts_model': self.tts_model,
                'tts_rate': self.tts_rate,
                'tts_pitch': self.tts_pitch,
                'tts_mode': self.tts_mode,
                'asr_language': self.asr_language,
                'asr_model': self.asr_model,
                'asr_mode': self.asr_mode,
                'whatsapp_mode_active': self.whatsapp_mode_active,
                'slack_mode_active': self.slack_mode_active,
                'voice_mode_active': self.voice_mode_active,
                'tap_to_talk_notification_shown': self.tap_to_talk_notification_shown,
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
    
    def get_in_tool_use(self) -> bool:
        """Thread-safe getter for in_tool_use"""
        with self._lock:
            return self.in_tool_use
    
    def set_in_tool_use(self, in_use: bool):
        """Set in_tool_use state (thread-safe)"""
        with self._lock:
            self.in_tool_use = in_use

    def get_mcp_server_running(self) -> bool:
        """Thread-safe getter for mcp_server_running"""
        with self._lock:
            return self.mcp_server_running

    def set_mcp_server_running(self, running: bool):
        """Set MCP server running state (thread-safe)"""
        with self._lock:
            old_value = self.mcp_server_running
            self.mcp_server_running = running
            changed = old_value != running

        if changed:
            self._trigger_callbacks('mcp_server_changed', running=running)


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
        """Load persisted state from file if available - only as defaults, not overriding explicit config"""
        try:
            import os
            state_file = os.path.expanduser('~/.talkito_state.json')
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    # Only load preferences that haven't been explicitly set
                    # Load provider info for background MCP processes
                    if 'tts_provider' in data:
                        self.state.tts_provider = data['tts_provider']
                    if 'asr_provider' in data:
                        self.state.asr_provider = data['asr_provider']
                    if 'tts_voice' in data:
                        self.state.tts_voice = data['tts_voice']
                    if 'tts_region' in data:
                        self.state.tts_region = data['tts_region']
                    if 'tts_language' in data:
                        self.state.tts_language = data['tts_language']
                    if 'tts_model' in data:
                        self.state.tts_model = data['tts_model']
                    if 'tts_rate' in data:
                        self.state.tts_rate = data['tts_rate']
                    if 'tts_pitch' in data:
                        self.state.tts_pitch = data['tts_pitch']
                    if 'tts_mode' in data:
                        self.state.tts_mode = data['tts_mode']
                    if 'asr_language' in data:
                        self.state.asr_language = data['asr_language']
                    if 'asr_model' in data:
                        self.state.asr_model = data['asr_model']
                    if 'asr_mode' in data:
                        self.state.asr_mode = data['asr_mode']
                    # One-time notification flags
                    if 'tap_to_talk_notification_shown' in data:
                        self.state.tap_to_talk_notification_shown = data['tap_to_talk_notification_shown']
                    # Communication config (these are persistent preferences)
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
                    if 'tap_to_talk_notification_shown' in data:
                        self.state.tap_to_talk_notification_shown = data['tap_to_talk_notification_shown']
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
                       model: Optional[str] = None, rate: Optional[float] = None,
                       pitch: Optional[float] = None):
        """Thread-safe TTS configuration update"""
        with self._lock:
            self.state.set_tts_config(provider, voice, region, language, model, rate, pitch)
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
    
    def set_asr_mode(self, mode: str):
        """Thread-safe ASR mode update"""
        with self._lock:
            self.state.asr_mode = mode
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


def initialize_providers_early(args):
    """Initialize providers early to trigger availability checks and download prompts"""
    start_time = time.time()
    log_message("INFO", f"initialize_providers_early called with args: {type(args)} [start_time={start_time}]")
    
    # Clear any cached provider validation results for fresh start
    asr_module = None
    try:
        asr_module = _import_asr()
        if hasattr(asr_module, 'clear_provider_cache'):
            asr_module.clear_provider_cache()
    except Exception:
        pass

    from . import tts

    shared_state = get_shared_state()

    # Capture desired ASR mode before initialization logic potentially overrides it
    requested_asr_mode = getattr(args, 'asr_mode', None) if hasattr(args, 'asr_mode') else None

    # Check if TTS is disabled before attempting provider initialization
    tts_disabled = (
        getattr(args, 'disable_tts', False) or
        getattr(args, 'tts_mode', None) == 'off' or
        os.environ.get('TALKITO_PREFERRED_TTS_PROVIDER') == 'off'
    )

    if tts_disabled:
        log_message("INFO", "Skipping TTS provider initialization - TTS mode is 'off'")
        shared_state.tts_provider = None
    else:
        # Trigger TTS provider selection (this will run availability checks and downloads)
        # Check both command line args and environment variable (same as ASR)
        step_start = time.time()
        requested_tts_provider = None
        if hasattr(args, 'tts_provider') and args.tts_provider:
            requested_tts_provider = args.tts_provider
            log_message("INFO", f"Found tts_provider in args: {requested_tts_provider}")
        preferred = requested_tts_provider or os.environ.get('TALKITO_PREFERRED_TTS_PROVIDER')
        log_message("INFO", f"Starting TTS provider selection for: {preferred}")
        try:
            if preferred:
                # Set the requested provider in shared state so select_best_tts_provider knows what was requested
                shared_state.tts_provider = preferred
            selected_tts = tts.select_best_tts_provider()

            if selected_tts is None:
                log_message("WARNING", f"No TTS provider available. TTS will be disabled.")
                print("No fallback TTS provider available. TTS will be disabled.")
                shared_state.tts_provider = None
                # Disable TTS since no providers are available
                if hasattr(args, 'disable_tts'):
                    args.disable_tts = True
            else:
                shared_state.tts_provider = selected_tts
                log_message("INFO", f"TTS provider selection completed: {selected_tts} [{time.time() - step_start:.3f}s]")

                # Start preloading TTS models early for local providers
                if selected_tts in ['kokoro', 'kittentts']:
                    try:
                        preload_start = time.time()
                        log_message("INFO", f"Starting early TTS model preloading for: {selected_tts}")
                        tts.preload_local_model(selected_tts)
                        log_message("INFO", f"Early TTS model preloading started for {selected_tts} [{time.time() - preload_start:.3f}s]")
                    except Exception as preload_e:
                        log_message("ERROR", f"Early TTS model preloading failed for {selected_tts}: {preload_e}")

        except Exception as e:
            log_message("ERROR", f"TTS provider selection failed: {e}")
            pass  # Continue even if selection fails
    
    # Check if ASR is disabled before attempting provider initialization
    asr_disabled = (
        requested_asr_mode == 'off' or
        os.environ.get('TALKITO_PREFERRED_ASR_PROVIDER') == 'off'
    )

    if asr_disabled:
        log_message("INFO", "Skipping ASR provider initialization - ASR mode is 'off'")
        shared_state.asr_provider = None
    else:
        # Trigger ASR provider selection (this will run availability checks and downloads)
        # Check both command line args and environment variable (claude command sets env var)
        step_start = time.time()
        requested_asr_provider = None
        if hasattr(args, 'asr_provider') and args.asr_provider:
            requested_asr_provider = args.asr_provider
            log_message("INFO", f"Found asr_provider in args: {requested_asr_provider}")
        preferred = requested_asr_provider or os.environ.get('TALKITO_PREFERRED_ASR_PROVIDER')
        log_message("INFO", f"Starting ASR provider selection for: {preferred}")
        if asr_module is None:
            asr_module = _import_asr()
        shared_state.asr_provider = asr_module.select_best_asr_provider(preferred)
        log_message("INFO", f"ASR provider selection completed: {shared_state.asr_provider} [{time.time() - step_start:.3f}s]")

        # Start preloading local whisper models early for local_whisper provider
        if shared_state.asr_provider == 'local_whisper':
            if requested_asr_mode == 'off':
                log_message("INFO", "Skipping PyWhisper preload - ASR mode requested as 'off'")
            else:
                try:
                    preload_start = time.time()
                    log_message("INFO", f"Starting early ASR model preloading for: {shared_state.asr_provider}")
                    asr_module.preload_local_asr_model()
                    log_message("INFO", f"Early ASR model preloading started for {shared_state.asr_provider} [{time.time() - preload_start:.3f}s]")
                except Exception as preload_e:
                    log_message("ERROR", f"Early ASR model preloading failed for {shared_state.asr_provider}: {preload_e}")
        else:
            log_message("INFO", f"Skipping PyWhisper preload - ASR provider is: {shared_state.asr_provider}")
    
    log_message("INFO", "initialize_providers_early about to complete")
    total_time = time.time() - start_time
    log_message("INFO", f"initialize_providers_early completed [total_time={total_time:.3f}s]")


def get_status_summary(comms_manager=None, whatsapp_recipient=None, slack_channel=None,
                       tts_override=False, asr_override=False, 
                       configured_tts_provider=None, configured_asr_provider=None) -> str:
    """Generate a one-line status summary for TalkiTo components with optional communication manager, recipients, channels, and provider overrides."""
    try:
        # Import needed modules
        from . import tts
        asr_module = _import_asr()
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
                "mode": shared_state.tts_mode,
                "provider": configured_tts_provider or shared_state.get_tts_provider() or "system",
                "voice": tts.get_tts_config().get('voice')
            },
            "asr": {
                "available": True,
                "initialized": shared_state.get_asr_initialized(),
                "enabled": shared_state.get_asr_enabled(),
                "mode": shared_state.asr_mode,
                "is_listening": asr_module.is_dictation_active() if shared_state.get_asr_initialized() else False,
                "provider": configured_asr_provider or shared_state.get_asr_provider() or "google"
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
        tts_display = f"TTS {status['tts']['mode']}"
        # Always show red if TTS mode is 'off', regardless of override
        if status['tts']['mode'] == 'off':
            tts_emoji = "🔴"
        elif tts_override or (status["tts"]["initialized"] and status["tts"]["enabled"]):
            tts_emoji = "🟢"
            tts_display += f" → {status['tts']['provider']}"
            if status['tts']['voice']:
                tts_display += f" {status['tts']['voice']}"
        else:
            tts_emoji = "🔴"
        asr_display = f"ASR {status['asr']['mode']}"
        if asr_override or (status["asr"]["initialized"] and status["asr"]["enabled"]):
            asr_emoji = "🟢"
            asr_display += f" → {status['asr']['provider'] or 'none'}"
            # Add tap-to-talk button indicator for supported modes
            if status['asr']['mode'] in ['tap-to-talk', 'push-to-talk']:
                asr_display += " (`)"
        else:
            asr_emoji = "🔴"
        
        # MCP server status
        mcp_running = shared_state.get_mcp_server_running()
        mcp_status = "🟢 MCP" if mcp_running else "🔴 MCP"

        # Communication status - check both configuration and mode status
        comms = []

        # Check if providers are configured (regardless of mode)
        if comms_manager:
            has_whatsapp = status["whatsapp"]["configured"]
            has_slack = status["slack"]["configured"]
        else:
            # Check from shared state if comm_manager not available
            # For WhatsApp: need both recipient number AND enabled flag
            has_whatsapp = shared_state.communication.whatsapp_enabled and shared_state.communication.whatsapp_to_number is not None
            # For Slack: need both channel AND enabled flag
            has_slack = shared_state.communication.slack_enabled and shared_state.communication.slack_channel is not None

        # Show status based on configuration and mode
        if has_whatsapp:
            if shared_state.get_whatsapp_mode_active():
                comms.append("🟢 WhatsApp")
            else:
                comms.append("⚪ WhatsApp")  # Configured but mode not active

        if has_slack:
            if shared_state.get_slack_mode_active():
                comms.append("🟢 Slack")
            else:
                comms.append("⚪ Slack")  # Configured but mode not active

        return f"TalkiTo: {tts_emoji} {tts_display} | {asr_emoji} {asr_display} | {mcp_status} | Comms: {', '.join(comms) if comms else 'none'}"
        
    except Exception as e:
        return f"Error getting status: {str(e)}"


def show_tap_to_talk_notification_once():
    """Show one-time notification about tap-to-talk mode change, if not already shown."""
    shared_state = get_shared_state()
    
    # Check if we've already shown this notification
    with shared_state._lock:
        if shared_state.tap_to_talk_notification_shown:
            return
        
        # Mark as shown and persist
        shared_state.tap_to_talk_notification_shown = True
    
    # Show the notification
    print("""
TALKITO DEFAULT ASR MODE HAS CHANGED
* The default ASR mode has changed to 'tap-to-talk' for better control.
* Hold the backtick key (`) to toggle voice input.
* You can change this back with: talkito --asr-mode auto-input or typing \"Switch to always on voice mode\"
""")

    save_shared_state()


def set_tts_config_thread_safe(provider: Optional[str] = None, voice: Optional[str] = None,
                               region: Optional[str] = None, language: Optional[str] = None,
                               model: Optional[str] = None, rate: Optional[float] = None,
                               pitch: Optional[float] = None):
    """Thread-safe helper to update TTS configuration"""
    _shared_state.set_tts_config(provider, voice, region, language, model, rate, pitch)


def set_asr_config_thread_safe(provider: Optional[str] = None, language: Optional[str] = None,
                               model: Optional[str] = None):
    """Thread-safe helper to update ASR configuration"""
    _shared_state.set_asr_config(provider, language, model)


def set_asr_mode_thread_safe(mode: str):
    """Thread-safe helper to update ASR mode"""
    _shared_state.set_asr_mode(mode)
