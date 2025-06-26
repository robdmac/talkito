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

ASR_LANGUAGE=en-US

# ASSEMBLYAI_API_KEY=your-assemby-key

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