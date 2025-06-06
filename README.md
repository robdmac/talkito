# TalkiTo

TalkiTo lets developers interact with AI systems through speech across multiple channels (terminal, API, phone). It can be used as both a command-line tool and a Python library.

## Installation

### From PyPI

```bash
# Basic installation
pip install talkito

# With ASR support
pip install talkito[asr]

# With all optional features
pip install talkito[all]
```

### From Source

```bash
# Clone the repository
git clone https://github.com/robdmac/talkito.git
cd talkito

# Install in development mode
pip install -e .

# Or install with optional dependencies
pip install -e ".[asr]"  # ASR support
pip install -e ".[all]"  # All features

# Make talkito command available (if not using pip install):
chmod +x talkito.sh
sudo ln -s $PWD/talkito.sh /usr/local/bin/talkito
```

## Usage

### Command-Line Usage

The primary way to use this tool is through the `talkito` command, which wraps any command and speaks its output:

```bash
# Basic usage - speaks command output
talkito echo "Hello, World!"

# Use with interactive programs
talkito python
talkito claude  # If you have Claude CLI installed
talkito aider   # If you have Aider installed

# MySQL with spoken output
talkito mysql -u root -p

# Long-running commands
talkito npm run dev
talkito python manage.py runserver
```

#### Advanced Options

```bash
# Auto-skip to newer content when text is long
talkito --auto-skip-tts claude

# Use different TTS providers
talkito --tts-provider openai --tts-voice nova echo "Hello with OpenAI"
talkito --tts-provider polly --tts-voice Matthew --tts-region us-west-2 echo "Hello with AWS"
talkito --tts-provider azure --tts-voice en-US-JennyNeural echo "Hello with Azure"
talkito --tts-provider gcloud --tts-voice en-US-Journey-F echo "Hello with Google"

# Use different ASR providers
talkito --asr-provider gcloud --asr-language en-US claude
talkito --asr-language es-ES echo "Hola mundo"  # Spanish recognition

# Enable remote communication
talkito --comms sms --sms-recipients +1234567890 long-running-command
talkito --comms slack python manage.py runserver
talkito --comms sms,whatsapp,slack server-monitor.sh
```

#### TTS Controls

While talkito.py is running:
- **Left Arrow (←)**: Restart current speech item
- **Right Arrow (→)**: Skip to next speech item

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

### Library Usage

Talkito can also be used as a Python library for integrating TTS and ASR into your applications:

```python
import asyncio
import talkito

# Simple command execution with TTS
async def run_with_speech():
    exit_code = await talkito.run_with_talkito(
        ['echo', 'Hello from Python!'],
        tts_config={'voice': 'nova'}
    )
    return exit_code

# Use TTS functionality directly
def speak_text():
    engine = talkito.detect_tts_engine()
    talkito.start_tts_worker(engine)
    talkito.queue_for_speech("This is a test")
    talkito.wait_for_tts_to_finish()
    talkito.shutdown_tts()

# Run the async example
asyncio.run(run_with_speech())
```

See the `examples/` directory for more detailed examples of library usage.

### MCP Server Usage

Talkito includes an MCP (Model Context Protocol) server that allows AI applications to use TTS and ASR capabilities:

```bash
# Install with MCP support
pip install talkito[mcp]

# Run as MCP server
talkito --mcp-server
```

The MCP server provides tools for:
- Text-to-Speech: `speak_text`, `skip_current_speech`, `configure_tts`
- Speech Recognition: `start_voice_input`, `stop_voice_input`, `get_dictated_text`
- Status monitoring: `get_speech_status`, `get_voice_input_status`

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
- **Usage**: `--comms sms --sms-recipients +1234567890`

#### Twilio WhatsApp
- **Get Started**: https://www.twilio.com/whatsapp
- **Setup**:
  -  Set Twilio credentials and `TWILIO_WHATSAPP_NUMBER`.
  - If you are messaging just yourself you can join the Twilio WhatsApp Sandbox.
  - Go to: https://www.twilio.com/console/sms/whatsapp/sandbox.
  - Follow the instructions to send a join code via WhatsApp to +1 415 523 8886.
  - Add the number provided in `TWILIO_WHATSAPP_NUMBER`
  - **important** you need to reply to the whatsapp message to enable the required free form (non templated) communications.
- **Usage**: `--comms whatsapp --whatsapp-recipients +1234567890`

#### Slack
- **Create App**: https://api.slack.com/apps
- **Setup**: Set `SLACK_BOT_TOKEN` and optionally `SLACK_APP_TOKEN`
- **Features**: Send output to channels, receive commands
- **Usage**: `--comms slack`

### Environment Configuration

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
# Edit .env with your API keys
```

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
