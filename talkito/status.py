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
status.py - Status reporting functionality for talkito
Centralizes status generation to avoid circular dependencies
"""

from typing import Optional, List, Dict, Any
from .state import get_shared_state


def get_status_summary(comms_manager=None, tts_override: Optional[bool] = None, asr_override: Optional[bool] = None) -> str:
    """Generate a status summary for talkito modules.
    
    This function was moved from state.py to avoid circular dependencies.
    
    Args:
        comms_manager: Optional communication manager instance
        tts_override: Override TTS status (for cases where we know it will be enabled)
        asr_override: Override ASR status (for cases where we know it will be enabled)
    
    Returns:
        Status summary string
    """
    status_parts = []
    
    # Lazy imports to avoid circular dependencies
    from . import tts
    from . import asr
    
    shared_state = get_shared_state()
    
    # TTS Status
    if tts_override is not None:
        tts_enabled = tts_override
    else:
        tts_enabled = shared_state.tts_enabled
    
    if tts_enabled:
        provider = shared_state.tts_provider or tts.tts_provider
        if provider and provider != 'system':
            status_parts.append(f"TTS: {provider}")
        else:
            # Check if system TTS is available
            system_engine = tts.detect_tts_engine()
            if system_engine != 'none':
                status_parts.append(f"TTS: {system_engine}")
            else:
                status_parts.append("TTS: off")
    else:
        status_parts.append("TTS: off")
    
    # ASR Status
    if asr_override is not None:
        asr_enabled = asr_override
    else:
        asr_enabled = shared_state.asr_enabled
    
    if asr_enabled and asr:
        provider = shared_state.asr_provider
        if provider and provider != 'google':
            status_parts.append(f"ASR: {provider}")
        else:
            # Check if we have better providers available
            try:
                best_provider = asr.select_best_asr_provider()
                if best_provider and best_provider != 'google':
                    status_parts.append(f"ASR: {best_provider}")
                else:
                    status_parts.append("ASR: google")
            except:
                status_parts.append("ASR: google")
    else:
        status_parts.append("ASR: off")
    
    # Communication Status - collect available providers
    available_comms = []
    
    # Check Twilio SMS
    from . import comms
    if comms.TWILIO_AVAILABLE:
        config = comms.create_config_from_env()
        if config.twilio_account_sid and config.twilio_phone_number:
            available_comms.append("SMS")
    
    # Check WhatsApp
    if comms.TWILIO_AVAILABLE:
        config = comms.create_config_from_env()
        if config.twilio_account_sid and config.twilio_whatsapp_number:
            available_comms.append("WhatsApp")
    
    # Check Slack
    if comms.SLACK_AVAILABLE:
        config = comms.create_config_from_env()
        if config.slack_bot_token and config.slack_app_token:
            available_comms.append("Slack")
    
    # Add communication status if any are available
    if available_comms:
        status_parts.append(f"Available Msgs: {', '.join(available_comms)}")
    
    # Format final status
    base_status = " | ".join(status_parts)
    
    # Add the note about .talkito.env if this looks like CLI startup
    if len(status_parts) >= 2:  # Has at least TTS and ASR
        return f"{base_status} (adjust with .talkito.env)"
    else:
        return base_status


def get_detailed_status() -> Dict[str, Any]:
    """Get detailed status information for all modules.
    
    Returns:
        Dictionary with detailed status for each module
    """
    from . import tts, asr, comms
    
    shared_state = get_shared_state()
    
    status = {
        'tts': {
            'enabled': shared_state.tts_enabled,
            'provider': shared_state.tts_provider or tts.tts_provider,
            'available_providers': list(tts.check_tts_provider_accessibility().keys()),
        },
        'asr': {
            'enabled': shared_state.asr_enabled,
            'provider': shared_state.asr_provider,
            'available_providers': [],
        },
        'communication': {
            'whatsapp_mode': shared_state.whatsapp_mode_active,
            'slack_mode': shared_state.slack_mode_active,
            'available_channels': [],
        }
    }
    
    # Check ASR providers if ASR is available
    if asr:
        try:
            # Get available ASR providers
            from . import asr as asr_module
            if hasattr(asr_module, 'check_asr_provider_accessibility'):
                status['asr']['available_providers'] = list(asr_module.check_asr_provider_accessibility().keys())
        except:
            pass
    
    # Check communication channels
    if comms.TWILIO_AVAILABLE or comms.SLACK_AVAILABLE:
        config = comms.create_config_from_env()
        
        if comms.TWILIO_AVAILABLE:
            if config.twilio_account_sid and config.twilio_phone_number:
                status['communication']['available_channels'].append('sms')
            if config.twilio_account_sid and config.twilio_whatsapp_number:
                status['communication']['available_channels'].append('whatsapp')
        
        if comms.SLACK_AVAILABLE:
            if config.slack_bot_token and config.slack_app_token:
                status['communication']['available_channels'].append('slack')
    
    return status