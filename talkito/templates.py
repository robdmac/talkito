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

"""Template files for Talkito initialization"""

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
