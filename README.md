# TalkiTo

TalkiTo lets developers interact with AI systems through speech across multiple channels (terminal, API, phone). It can be used as both a command-line tool and a Python library.

## Quick Start Guide using Claude

### From Source

```bash
# Clone the repository
git clone https://github.com/robdmac/talkito.git
cd talkito

pip install -e ".[all]"  # All features

talkito claude
```

### How It Works with Claude

When you run `talkito claude`, it automatically:

1. **Starts an MCP SSE server** in the background for real-time notifications
2. **Configures Claude** to connect to the talkito MCP server
3. **Falls back gracefully** if SSE isn't supported (SSE → stdio → traditional wrapper)

### Voice Mode

Once Claude is running with talkito, you can activate voice mode:

1. **Enable voice mode**: Use the `talkito:turn_on` or simply "talkito on" in Claude
2. **Continuous listening**: Claude will then automatically:
   - Speak its responses using TTS
   - Start listening for your voice input
   - Process your speech as the next command
   - Continue this loop until you say "stop voice mode"

3. **Unified input handling**: All messages are treated as user input:
   - Voice dictation: Processed as spoken commands
   - Slack messages: Processed as typed commands (when connected)
   - WhatsApp messages: Processed as typed commands (when connected)

### Real-Time Notifications

The SSE server enables Claude to receive notifications when:
- New voice input is available
- Messages arrive from Slack or WhatsApp
- You can use `talkito:start_notification_stream` to begin receiving updates

### Communication Channels

Configure environment variables to enable remote messaging:

```bash
# WhatsApp setup (run `talkito --setup-whatsapp` for full instructions)
export TWILIO_ACCOUNT_SID='your_sid'
export TWILIO_AUTH_TOKEN='your_token'
export TWILIO_WHATSAPP_NUMBER='whatsapp:+14155238886'

# Slack setup (run `talkito --setup-slack` for full instructions)
export SLACK_BOT_TOKEN='xoxb-...'
export SLACK_APP_TOKEN='xapp-...'
export SLACK_CHANNEL='#talkito-dev'
```

Then in Claude voice mode:
- Say "slack me at #channel" to enable Slack mode
- Say "whatsapp me at +1234567890" to enable WhatsApp mode
- All your responses will be automatically sent to the configured channel

## Usage

### Command-Line Usage

The primary way to use this tool is through the `talkito` command, which wraps any command and speaks its output:

```bash
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
