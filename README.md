# TalkiTo

<div align="center">

[![GitHub Stars](https://img.shields.io/github/stars/robdmac/talkito?style=social)](https://github.com/robdmac/talkito/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/robdmac/talkito?style=social)](https://github.com/robdmac/talkito/network/members)
[![License](https://img.shields.io/badge/License-AGPL%20v3-blue.svg?style=flat-square)](https://github.com/robdmac/talkito/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1420523410513072198?style=flat-square&logo=discord&logoColor=white)](https://discord.gg/WbP58Tym)

</div>

TalkiTo lets developers talk, slack and whatsapp with Claude Code and OpenAI Codex. It can be used as a command-line tool, a web extension, and as a Python library.

## 🚀 Quick Install

### Option 1: One-liner Install Script (Recommended)
```bash
curl -sSL https://raw.githubusercontent.com/robdmac/talkito/main/install.sh | bash
```

### Option 2: PyPI (Coming Soon)
```bash
pip install talkito
```

Then just run:
```bash
talkito claude
```

## Install for End Users

### From Source (Stable)
```bash
# Clone the repository
git clone https://github.com/robdmac/talkito.git
cd talkito

# Create and activate virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install system dependencies (macOS)
brew install portaudio

# Install package (normal install - gets updates via git pull)
pip install .

# Run this in a directory you want to use claude with
talkito claude
```

## Install for Developers

### Editable Install (Development)
```bash
# Clone the repository
git clone https://github.com/robdmac/talkito.git
cd talkito

# Create and activate virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install system dependencies (macOS)
brew install portaudio

# Install in development mode (editable install)
pip install -e .

# Run this in a directory you want to use claude with
talkito claude
```

or for the web extension run as
```commandline
talkito --mcp-sse-server
```
then go to chrome://extensions/ and load unpacked the extensions/chrome/ dir

## Demo Video

[![TalkiTo Demo](https://img.youtube.com/vi/FJdYTYZK_0U/0.jpg)](https://youtu.be/FJdYTYZK_0U)

## AI Assistant Compatibility

| AI Assistant                 | Method        | Status              |
|------------------------------|---------------|---------------------|
| **Claude Code**              | Terminal      | **Fully Supported** |
| bolt.new                     | Web Extension | Output Only         |
| v0.dev                       | Web Extension | Output Only         |
| replit.com                   | Web Extension | Output Only         |
| Gemini CLI                   | Terminal      | In Progress         |
| Codex                        | Terminal      | In Progress         |
| Aider                        | Terminal      | In Progress         |
| Cursor                       | Terminal      | In Progress         |
| Continue                     | Terminal      | In Progress         |



### Voice Mode

When you run `talkito claude`, voice mode is enabled by default:

1. **Automatic voice interaction**: Claude will:
   - Speak all responses using TTS
   - Listen for your voice input after speaking
   - Process your speech as the next user message
   - Continue this loop automatically

2. **Control voice mode**: 
   - Voice mode starts ON by default
   - Say or type "turn off talkito" to disable voice interaction
   - Say or type "turn on talkito" to re-enable if turned off

3. **Unified input handling**: All inputs are processed as user messages:
   - Voice dictation: Your spoken words
   - Slack messages: From configured channels
   - WhatsApp messages: From configured numbers

4. **Communication modes**: 
   - Say "start slack mode #channel-name" to auto-send responses to Slack
   - Say "start whatsapp mode +1234567890" to auto-send responses to WhatsApp
   - Say "stop slack/whatsapp mode" to disable


#### Advanced Options

```bash
# Disable auto-skip to newer content (auto-skip is on by default)
talkito --dont-auto-skip-tts claude

# Use different TTS providers
talkito --tts-provider openai --tts-voice nova echo "Hello with OpenAI"
talkito --tts-provider polly --tts-voice Matthew --tts-region us-west-2 echo "Hello with AWS"
talkito --tts-provider azure --tts-voice en-US-JennyNeural echo "Hello with Azure"
talkito --tts-provider gcloud --tts-voice en-US-Journey-F echo "Hello with Google"

# Use different ASR providers
talkito --asr-provider gcloud --asr-language en-US claude
talkito --asr-language es-ES echo "Hola mundo"  # Spanish recognition

# Enable remote communication (configure via environment variables)
talkito --slack-channel '#alerts' python manage.py runserver
talkito --whatsapp-recipients +1234567890 long-running-command
talkito --sms-recipients +1234567890,+0987654321 server-monitor.sh
```

### Using tts.py (Standalone TTS)

The TTS module can be used independently for text-to-speech operations:

```python
#!/usr/bin/env python3
import tts

# Initialize TTS
engine = tts.detect_tts_engine()
tts.start_tts_worker(engine)

# Speak text
tts.queue_for_speech("Hello from the TTS module!")

# Wait and cleanup
import time
time.sleep(2)
tts.shutdown_tts()
```

### Using asr.py (Standalone ASR)

The ASR module can be used independently for speech recognition:

```python
#!/usr/bin/env python3
import asr

# Define callback for recognized text
def handle_text(text):
    print(f"You said: {text}")

# Start dictation
asr.start_dictation(handle_text)

# Keep running (press Ctrl+C to stop)
try:
    import time
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    asr.stop_dictation()
```

### MCP Server Usage

Talkito includes an MCP (Model Context Protocol) server that allows AI applications to use TTS and ASR capabilities:

```bash
# Install TalkiTo (includes MCP support)
pip install talkito

# Run as MCP server
talkito --mcp-server
```

The MCP server provides tools for:
- **Core**: `turn_on`/`turn_off` (enable voice mode), `get_talkito_status`
- **TTS**: `enable_tts`/`disable_tts`, `speak_text`, `skip_current_speech`, `configure_tts`
- **ASR**: `enable_asr`/`disable_asr`, `start_voice_input`/`stop_voice_input`, `get_dictated_text`
- **Communication**: `start_whatsapp_mode`/`stop_whatsapp_mode`, `start_slack_mode`/`stop_slack_mode`, `send_whatsapp`, `send_slack`, `get_messages`

Configure your AI application to connect to the talkito MCP server for voice capabilities.

## Provider Configuration

### Text-to-Speech (TTS) Providers

#### System TTS (Default)
- **macOS**: Uses built-in `say` command
- **Linux**: Uses `espeak`, `festival`, or `flite` (install via package manager)
- **Setup**: No API key needed

#### OpenAI TTS
- **Get API Key**: https://platform.openai.com/api-keys
- **Voices**: alloy, echo, fable, onyx, nova, shimmer
- **Usage**: `--tts-provider openai --tts-voice nova`

#### AWS Polly
- **Get Credentials**: https://aws.amazon.com/polly/getting-started/
- **Setup**: Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
- **Voices**: Joanna, Matthew, Amy, Brian, and more
- **Usage**: `--tts-provider polly --tts-voice Matthew`

#### Azure Speech Services
- **Get API Key**: https://azure.microsoft.com/en-us/services/cognitive-services/speech-services/
- **Setup**: Set `AZURE_SPEECH_KEY` and `AZURE_REGION`
- **Voices**: en-US-JennyNeural, en-US-AriaNeural, and many more
- **Usage**: `--tts-provider azure --tts-voice en-US-JennyNeural`

#### Google Cloud Text-to-Speech
- **Get Credentials**: https://cloud.google.com/text-to-speech/docs/quickstart
- **Setup**: Set `GOOGLE_APPLICATION_CREDENTIALS` to service account JSON path
- **Voices**: en-US-Journey-F, en-US-News-N, and more
- **Usage**: `--tts-provider gcloud --tts-voice en-US-Journey-F`

#### ElevenLabs
- **Get API Key**: https://elevenlabs.io/
- **Setup**: Set `ELEVENLABS_API_KEY`
- **Voices**: Various voice IDs available
- **Usage**: Configure in code or .env file

#### Deepgram
- **Get API Key**: https://deepgram.com/
- **Setup**: Set `DEEPGRAM_API_KEY`
- **Voices**: aura-asteria-en, aura-luna-en, aura-stella-en, and more
- **Usage**: `--tts-provider deepgram --tts-voice aura-asteria-en`

### Automatic Speech Recognition (ASR) Providers

#### Google Speech Recognition (Default)
- **Free**: No API key required
- **Limitations**: Best for short utterances, requires internet
- **Usage**: Default when no provider specified

#### Google Cloud Speech-to-Text
- **Get Credentials**: https://cloud.google.com/speech-to-text/docs/quickstart
- **Setup**: Set `GOOGLE_APPLICATION_CREDENTIALS`
- **Features**: Better accuracy, streaming support
- **Usage**: `--asr-provider gcloud`

#### AssemblyAI
- **Get API Key**: https://www.assemblyai.com/
- **Setup**: Set `ASSEMBLYAI_API_KEY`
- **Features**: Real-time transcription, speaker detection
- **Usage**: Configure in code or .env file

#### Deepgram
- **Get API Key**: https://deepgram.com/
- **Setup**: Set `DEEPGRAM_API_KEY`
- **Features**: Fast, accurate real-time transcription
- **Usage**: Configure in code or .env file

#### Houndify
- **Get Credentials**: https://www.houndify.com/
- **Setup**: Set `HOUNDIFY_CLIENT_ID` and `HOUNDIFY_CLIENT_KEY`
- **Features**: Natural language understanding
- **Usage**: `--asr-provider houndify`

#### AWS Transcribe
- **Get Credentials**: https://aws.amazon.com/transcribe/
- **Setup**: Set AWS credentials
- **Features**: Streaming transcription
- **Usage**: `--asr-provider aws --aws-region us-west-2`

### Communication Providers (Remote Interaction)

#### Twilio SMS
- **Get Account**: https://www.twilio.com/try-twilio
- **Setup**: Set `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER` you will need to a verified number to avoid being filtered.
- **Features**: Send command output via SMS, receive input via SMS
- **Usage**: `--sms-recipients +1234567890`

#### Twilio WhatsApp
- **Get Started**: https://www.twilio.com/whatsapp
- **Setup Instructions**: Run `talkito --setup-whatsapp` for detailed setup guide
- **Required Environment Variables**:
  - `TWILIO_ACCOUNT_SID`: Your Twilio account SID
  - `TWILIO_AUTH_TOKEN`: Your Twilio auth token
  - `TWILIO_WHATSAPP_NUMBER`: Twilio's WhatsApp number (usually +14155238886)
  - `WHATSAPP_RECIPIENTS`: Your WhatsApp number
  - `ZROK_RESERVED_TOKEN`: Your zrok reserved share token
- **Quick Setup**:
  - Join Twilio WhatsApp Sandbox at https://www.twilio.com/console/sms/whatsapp/sandbox
  - Send the join code via WhatsApp to +1 415 523 8886
  - Install zrok and create a reserved share: `zrok reserve public http://localhost:8080`
  - Set webhook URL in Twilio Console to: `https://YOUR-TOKEN.share.zrok.io/whatsapp`
- **Usage**: `--whatsapp-recipients +1234567890`

#### Slack
- **Create App**: https://api.slack.com/apps
- **Setup**: Set `SLACK_BOT_TOKEN` and optionally `SLACK_APP_TOKEN`
- **Features**: Send output to channels, receive commands
- **Usage**: `--slack-channel '#channel-name'`

### Environment Configuration

Talkito supports two environment files:
- `.env` - Primary configuration (takes precedence)
- `.talkito.env` - Secondary configuration (won't override `.env`)

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
# Edit .env with your API keys
```

For WhatsApp setup with zrok tunneling:
- `ZROK_RESERVED_TOKEN`: Your zrok reserved share token for webhook tunneling

## Requirements

- Python 3.8+
- macOS (with `say` command) or Linux (with `espeak`, `festival`, or `flite`)
- Optional: `SpeechRecognition` and `pyaudio` for ASR support
- Optional: Provider-specific Python packages (installed as needed)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the GNU Affero General Public License v3.0 or later - see the [LICENSE](LICENSE) file for details.

Copyright (C) 2025 Robert Macrae
