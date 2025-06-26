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
Template files for Talkito initialization
"""

ENV_EXAMPLE_TEMPLATE = """# Talkito environment configuration template
# Copy settings from this file to your .env file and fill in the API keys you want to use

# Preferred Providers
# When no --tts-provider or --asr-provider flags are specified, talkito will:
# 1. Use these preferred providers if they're accessible
# 2. Otherwise, use the first accessible non-default provider (alphabetically)
# 3. Fall back to system TTS or google ASR if no other providers are available

# TALKITO_PREFERRED_TTS_PROVIDER=openai  # Options: system, openai, aws, polly, azure, gcloud, elevenlabs, deepgram
# TALKITO_PREFERRED_ASR_PROVIDER=assemblyai  # Options: google, gcloud, assemblyai, deepgram, openai, azure, aws, bing

ASR_LANGUAGE=en-US

# Unified VAD Configuration
# Minimum silence duration (in milliseconds) before ending utterance
# This affects all ASR providers to make them less aggressive about cutting off speech
# Default: 1500ms (1.5 seconds)
# TALKITO_ASR_MIN_SILENCE_MS=1500

# ASSEMBLYAI_API_KEY=your-assemby-key
# ASSEMBLYAI_END_OF_TURN_CONFIDENCE=0.5  # Lower = less aggressive (default: 0.5, range: 0.0-1.0)
# ASSEMBLYAI_MAX_SILENCE_MS=5000  # Maximum silence before end-of-turn (default: 5000ms)

# AWS_ACCESS_KEY_ID=your-aws-access-key-id
# AWS_SECRET_ACCESS_KEY=your-aws-secret-access-key
# AWS_DEFAULT_REGION=us-east-1
# AWS_POLLY_VOICE=Joanna

# AZURE_SPEECH_KEY=your-azure-speech-key-here
# AZURE_VOICE=en-US-AriaNeural
# AZURE_REGION=eastus

# BING_KEY=your-bing-key

# DEEPGRAM_API_KEY=your-deepgram-key
# DEEPGRAM_VOICE_MODEL=aura-asteria-en

# ELEVENLABS_API_KEY=your-elevenlabs-api-key-here
# ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
# ELEVENLABS_MODEL_ID=eleven_monolingual_v1

# GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/google-cloud-service-account.json
# GCLOUD_VOICE=en-US-Journey-F
# GCLOUD_LANGUAGE_CODE=en-US

# HOUNDIFY_CLIENT_ID=your-houndify-client-id
# HOUNDIFY_CLIENT_KEY=your-houndify-client-key

# OPENAI_API_KEY=your-openai-api-key-here
# OPENAI_VOICE=alloy

# Communication Provider Settings (for remote interaction)

# Twilio (for SMS and WhatsApp)
# TWILIO_ACCOUNT_SID=your-twilio-account-sid
# TWILIO_AUTH_TOKEN=your-twilio-auth-token
# TWILIO_PHONE_NUMBER=+1234567890  # Your Twilio phone number for SMS
# TWILIO_WHATSAPP_NUMBER=+1234567890  # Your Twilio WhatsApp number

# SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
# SLACK_CHANNEL=#talkito  # Default channel for messages
# SLACK_APP_TOKEN=xapp-your-slack-app-token  # For socket mode (optional)

# ZROK_RESERVED_TOKEN
"""

TALKITO_MD_CONTENT = """# Talkito Voice Interaction Instructions

## Voice Mode Control

### When `talkito:turn_on` is called:
Enter voice interaction mode and follow these patterns for EVERY interaction:

1. **Acknowledge activation**: Confirm voice mode is active
2. **Continuous voice loop**:
   - After EVERY response, automatically call `talkito:speak_text` with your key conclusion
   - Immediately call `talkito:start_voice_input` without asking
   - Wait 1 second, then silently call `talkito:get_messages`
   - Process the received messages as the next user input
   - Continue this cycle indefinitely while talkito is on

3. **Natural conversation flow**:
   - Keep spoken responses concise (1-2 sentences max)
   - Speak naturally, as in a voice conversation
   - Don't mention that you're calling tools
   - Make it seamless and hands-free

### When `talkito:turn_off` is called:
1. Exit voice interaction mode immediately
2. Confirm deactivation with a final speech output
3. Return to standard text-only interaction
4. Stop all automatic speech and voice input

### Voice Commands to Listen For:
- "stop voice mode" or "turn off voice" → automatically call `talkito:turn_off`
- "pause listening" → skip the next voice input cycle
- "repeat that" → speak the last response again
- "whatsapp me at [phone number]" → call `talkito:start_whatsapp_mode` with the extracted phone number
- "start whatsapp mode" → call `talkito:start_whatsapp_mode` (uses TWILIO_WHATSAPP_NUMBER)
- "stop whatsapp mode" → call `talkito:stop_whatsapp_mode`
- "slack me at [channel]" → call `talkito:start_slack_mode` with the extracted channel
- "start slack mode" → call `talkito:start_slack_mode` (uses SLACK_CHANNEL)
- "stop slack mode" → call `talkito:stop_slack_mode`
- "send to WhatsApp" → use `talkito:send_whatsapp` for the last response
- "send to Slack" → use `talkito:send_slack` for the last response

## Standard Mode (default or after turn_off)
- Only use talkito tools when explicitly requested
- No automatic speech output
- No automatic voice input
- Normal Claude text interaction

## Communication Channels

### WhatsApp Integration:
- Use `talkito:send_whatsapp` to send messages via WhatsApp
- Can combine with TTS: set `with_tts=true` to also speak the message
- Configure with `talkito:configure_communication` or via TWILIO environment variables

### Slack Integration:
- Use `talkito:send_slack` to send messages to Slack channels
- Can combine with TTS: set `with_tts=true` to also speak the message
- Configure with `talkito:configure_communication` or via SLACK_BOT_TOKEN

### Communication Status:
- Check channel availability with `talkito:get_communication_status`
- View configuration in resource: `talkito://communication/status`

### WhatsApp Mode:
When WhatsApp mode is active:
- All your responses are automatically sent to WhatsApp
- Voice commands trigger immediate WhatsApp messages
- Perfect for hands-free messaging while driving or cooking
- Use `talkito:start_whatsapp_mode` to activate
- Use `talkito:stop_whatsapp_mode` to deactivate
- Check status with `talkito:get_whatsapp_mode_status`

### Slack Mode:
When Slack mode is active:
- All your responses are automatically sent to a Slack channel
- Voice commands trigger immediate Slack messages
- Great for team collaboration and hands-free updates
- Use `talkito:start_slack_mode` to activate
- Use `talkito:stop_slack_mode` to deactivate
- Check status with `talkito:get_slack_mode_status`

## Provider Selection

### TTS Providers:
- Automatic selection prefers non-system providers (aws, openai, azure, etc.)
- Set TALKITO_PREFERRED_TTS_PROVIDER to choose default
- 'polly' is now an alias for 'aws' (both work identically)

### ASR Providers:
- Automatic selection prefers authenticated providers over free Google
- Set TALKITO_PREFERRED_ASR_PROVIDER to choose default
- Voice activity detection is less aggressive (1.5s silence threshold)
- Unified setting: TALKITO_ASR_MIN_SILENCE_MS (default: 1500ms)

## Message Handling

### Unified Message Retrieval:
- Use `talkito:get_messages` to check all input sources at once
- Returns messages from voice dictation, Slack, and WhatsApp in one call
- Messages are cleared after reading to avoid duplicates
- Check frequently when in voice mode or when expecting responses


## Important Notes:
- In voice mode, ALWAYS continue the cycle until explicitly stopped
- Never ask permission to start voice input in voice mode
- Keep the interaction natural and conversational
- If voice input returns empty/None, try again after a brief pause
- VAD settings are optimized for natural speech with pauses
"""

SLACK_BOT_MANIFEST = """
{
    "display_information": {
        "name": "TalkiTo",
        "description": "Voice-enabled command execution via TalkiTo"
    },
    "features": {
        "bot_user": {
            "display_name": "TalkiTo",
            "always_online": false
        },
        "slash_commands": [
            {
                "command": "/talkito",
                "description": "Send commands to TalkiTo",
                "usage_hint": "[command]",
                "should_escape": false
            }
        ]
    },
    "oauth_config": {
        "scopes": {
            "bot": [
                "channels:history",
                "chat:write",
                "groups:history",
                "im:history",
                "mpim:history",
                "channels:read"
            ]
        }
    },
    "settings": {
        "event_subscriptions": {
            "bot_events": [
                "message.channels",
                "message.groups",
                "message.im",
                "message.mpim"
            ]
        },
        "interactivity": {
            "is_enabled": true
        },
        "org_deploy_enabled": false,
        "socket_mode_enabled": true,
        "token_rotation_enabled": false
    }
}
"""
