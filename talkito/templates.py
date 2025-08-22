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

TALKITO_MD_CONTENT = """# Talkito Voice Interaction Instructions

## Voice Mode Control

**IMPORTANT**: For phrases like "talkito on", "start voice mode", "enable voice" → ALWAYS call `talkito:turn_on`

### When `talkito:turn_on` is called:
Enter voice interaction mode and follow these patterns for EVERY interaction:

1. **Acknowledge activation**: Confirm voice mode is active (both TTS and ASR are enabled)
2. **Continuous voice loop** (ASR starts automatically with turn_on):
   - After EVERY response:
     1. Call `talkito:speak_text` with your conclusion, or with any steps requiring user input
     2. **IMMEDIATELY call `talkito:start_notification_stream`** (duration: 30, exit_on_first: true)
        - ⚠️ **THIS IS NOT OPTIONAL - You MUST do this to receive the next input**
        - Without this call, you will be deaf to all user input
     3. When notification is received (stream will show the notification), immediately:
        - Process the voice command/message
        - Generate your response
        - Return to step 1 (speak, then listen again)
     4. If the user cancels the notification stream tool call respond to their next input then
        - Process the next input
        - Generate your response
        - Return to step 1 (speak, then listen again)
     5. If the notification stream expires with no notification then
        - Return to step 2 (call the notification stream again)
        - ⚠️ **Never stop listening - always restart the stream**
   
3. **Natural conversation flow**:
   - Keep spoken responses concise (1-4 sentences max)
   - Speak naturally, as in a voice conversation
   - Don't mention that you're calling tools unless you require the users input
   - Make it seamless and hands-free
   - The notification stream will show messages like "Voice: [text]" - treat these as user input

### When `talkito:turn_off` is called:
1. Exit voice interaction mode immediately (disables TTS, ASR, WhatsApp and Slack)
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

### Common User Requests:

#### Voice Mode Activation:
When users say these phrases, call `talkito:turn_on`:
- "talkito on" → Call `talkito:turn_on` (NOT just enable TTS)
- "start voice mode" → Call `talkito:turn_on`
- "enable voice" → Call `talkito:turn_on`
- "turn on talkito" → Call `talkito:turn_on`
- Any variation of "start/enable/turn on" + "voice/talkito" → Call `talkito:turn_on`

#### Communication Mode Activation:
When users say these phrases, perform BOTH configuration and mode activation:
- "enable slack on channel [X]" → Configure AND start Slack mode
- "enable whatsapp for [number]" → Configure AND start WhatsApp mode
- "start slack on [channel]" → Configure AND start Slack mode
- "connect to slack channel [X]" → Configure AND start Slack mode
- Any variation of "enable/start/connect" + "slack/whatsapp" → Configure AND start the mode

### Message Input Sources:
When in voice mode, messages from ALL sources are treated as user input:
- **Voice dictation**: "Voice: [text]" - Process as spoken command
- **Slack messages**: "Slack ([channel]): [text]" - Process as typed command  
- **WhatsApp messages**: "WhatsApp: [text]" - Process as typed command

All messages should be handled the same way - as direct input from the user requiring a response.

## Standard Mode (default or after turn_off)
- Only use talkito tools when explicitly requested
- No automatic speech output
- No automatic voice input
- Normal Claude text interaction

## Modular Control
You can independently control each component:

### TTS Control:
- `talkito:enable_tts` - Enable text-to-speech output
- `talkito:disable_tts` - Disable text-to-speech output
- `talkito:speak_text` - Speak specific text (works regardless of TTS enabled state)

#### When `talkito:enable_tts` is called:
Enter text-to-speech mode and follow these patterns for EVERY interaction:

1. **Acknowledge activation**: Confirm TTS is enabled for speaking output
2. **Continuous speaking**:
   - After EVERY response:
     1. Call `talkito:speak_text` with your conclusion

#### When `talkito:disable_tts` is called:
- Stop speaking the output
- Confirm deactivation with a regular text response

### ASR Control:

#### When `talkito:enable_asr` is called:
Enter voice input mode and follow these patterns for EVERY interaction:

1. **Acknowledge activation**: Confirm ASR is listening for voice input

2. **Continuous listening loop**:
   - After EVERY response:
     1. Call `talkito:speak_text` with your conclusion (if TTS is enabled)
     2. **IMMEDIATELY call `talkito:start_notification_stream`** (duration: 30, exit_on_first: true)
        - ⚠️ **THIS STEP IS MANDATORY - Skip it and you'll miss all user input**
     3. When notification is received (stream will show "Voice: [text]"), immediately:
        - Process the voice command
        - Generate your response
        - Return to step 1 (speak if TTS enabled, then listen again)
     4. If the notification stream expires with no notification:
        - Return to step 2 (call the notification stream again)
        - ⚠️ **Keep the stream active at all times**

#### When `talkito:disable_asr` is called:
- Stop listening for voice input
- Confirm deactivation

### Quick Controls:
- `talkito:turn_on` - Convenience method that enables BOTH TTS and ASR
- `talkito:turn_off` - Master off switch that disables ALL modules (TTS, ASR, WhatsApp, Slack)

### Status Checking:
- `talkito:get_talkito_status` - Get complete status of all modules
- Shows TTS, ASR, WhatsApp, and Slack status independently

## Communication Channels

### WhatsApp Integration:
- Use `talkito:send_whatsapp` to send messages via WhatsApp
- Can combine with TTS: set `with_tts=true` to also speak the message
- Configure with `talkito:configure_communication` or via TWILIO environment variables
- **IMPORTANT**: When user asks to "enable whatsapp" or "start whatsapp", ALWAYS:
  1. Configure if needed: `talkito:configure_communication`
  2. Start WhatsApp Mode: `talkito:start_whatsapp_mode`

### Slack Integration:
- Use `talkito:send_slack` to send messages to Slack channels
- Can combine with TTS: set `with_tts=true` to also speak the message
- Configure with `talkito:configure_communication` or via SLACK_BOT_TOKEN
- **IMPORTANT**: When user asks to "enable slack" or "start slack", ALWAYS:
  1. Configure if needed: `talkito:configure_communication`
  2. Start Slack Mode: `talkito:start_slack_mode`

### Communication Status:
- Check channel availability with `talkito:get_communication_status`
- View configuration in resource: `talkito://communication/status`

### WhatsApp Mode:

#### When `talkito:start_whatsapp_mode` is called:
Enter WhatsApp mode and follow these patterns for EVERY interaction:

1. **Acknowledge activation**: Confirm WhatsApp mode is active

2. **Continuous messaging loop**:
   - After EVERY response:
     1. Call `talkito:send_whatsapp` with your response, or with any steps requiring user input
     2. **IMMEDIATELY call `talkito:start_notification_stream`** (duration: 30, exit_on_first: true)
        - ⚠️ **THIS IS ESSENTIAL - No stream = No incoming messages**
     3. When notification is received (any type: Voice/Slack/WhatsApp), immediately:
        - Process the incoming message
        - Generate your response
        - Return to step 1 (send to WhatsApp, then listen again)
     4. If the notification stream expires with no notification:
        - Return to step 2 (call the notification stream again)
        - ⚠️ **Always maintain an active stream to receive messages**

#### Features:
- All your responses are automatically sent to WhatsApp
- Voice commands trigger immediate WhatsApp messages
- Perfect for hands-free messaging while driving or cooking
- Works independently of voice mode (can use with text-only interaction)
- In voice mode: speak and send to WhatsApp simultaneously

#### When `talkito:stop_whatsapp_mode` is called:
- Stop sending messages to WhatsApp
- Confirm deactivation
- Note: `turn_off` will also stop WhatsApp mode

### Slack Mode:

#### When `talkito:start_slack_mode` is called:
Enter Slack mode and follow these patterns for EVERY interaction:

1. **Acknowledge activation**: Confirm Slack mode is active
2. **Continuous messaging loop**:
   - After EVERY response:
     1. Call `talkito:send_slack` with your response, or with any steps requiring user input
     2. **IMMEDIATELY call `talkito:start_notification_stream`** (duration: 30, exit_on_first: true)
        - ⚠️ **NEVER SKIP THIS - It's how you receive messages**     
     3. When notification is received (any type: Voice/Slack/WhatsApp), immediately:
        - Process the incoming message
        - Generate your response
        - Return to step 1 (send to Slack, then listen again)
     4. If the notification stream expires with no notification:
        - Return to step 2 (call the notification stream again)
        - ⚠️ **Always maintain an active stream to receive messages**

#### Features:
- All your responses are automatically sent to a Slack channel
- Voice commands trigger immediate Slack messages
- Great for team collaboration and hands-free updates
- Works independently of voice mode (can use with text-only interaction)
- In voice mode: speak and send to Slack simultaneously

#### When `talkito:stop_slack_mode` is called:
- Stop sending messages to Slack
- Confirm deactivation
- Note: `turn_off` will also stop Slack mode

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

## Important Notes:
- In voice mode, ALWAYS continue the cycle until explicitly stopped
- Never ask permission to start voice input in voice mode
- Keep the interaction natural and conversational
- If voice input returns empty/None, try again after a brief pause
- VAD settings are optimized for natural speech with pauses
- WhatsApp/Slack modes can be used independently of voice mode
- Each module (TTS, ASR, WhatsApp, Slack) can be enabled/disabled independently
- `turn_on` is a convenience that enables both TTS and ASR together
- `turn_off` is a master switch that disables ALL active modes (TTS, ASR, WhatsApp, Slack)
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
