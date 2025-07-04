# TalkiTo Chrome Extension

A Chrome extension that monitors AI coding websites (bolt.new, v0.dev, replit.com) and automatically speaks their outputs via TTS. Also supports sending content to WhatsApp and Slack, and includes a right-click context menu for any webpage.

## Features

### Automatic Monitoring (New!)
- **Auto-enabled on AI Sites**: Automatically monitors bolt.new, v0.dev, and replit.com
- **Toggle Controls**: Popup interface with master toggle and individual controls for:
  - TTS (Text-to-Speech) - Automatically enabled when monitoring starts
  - WhatsApp - Send outputs to WhatsApp
  - Slack - Send outputs to Slack
- **Smart Toggle Behavior**: 
  - Main toggle ON → TTS automatically enabled
  - Main toggle OFF → All services disabled

### Context Menu Features
- **Three Action Types**: Talk (TTS), WhatsApp, and Slack
- **Right-click Context Menu**: Submenu with three options: "Talk", "WhatsApp", and "Slack"
- **Live Text Monitoring**: Continuously monitors element text content changes
- **Server Discovery**: Automatically discovers server on ports 8000-8010
- **User Preferences**: Prompts for phone number (WhatsApp) and channel (Slack) on first use
- **Visual Indicators**: Different colored outlines for each action type
  - Red: Talk
  - Green: WhatsApp
  - Purple: Slack
- **AJAX Support**: Detects and sends text changes from dynamic content updates
- **Keyboard Shortcut**: Use Ctrl+Shift+L to stop monitoring
- **Fallback Logging**: Falls back to console logging if server is unavailable

## Server Requirements

The extension expects a server running on any port between 8000-8010 with these endpoints:

### Discovery
- GET `/api/ping` - Server health check

### Talk (TTS)
- POST `/api/speak` - Speak text via TTS
- Request: `{ "text": "text to speak" }`
- POST `/api/tts/enable` - Enable TTS
- POST `/api/tts/disable` - Disable TTS

### WhatsApp
- POST `/api/whatsapp` - Send WhatsApp message
- Request: `{ "message": "text", "to_number": "+1234567890", "with_tts": true }`
- POST `/api/whatsapp/start` - Start WhatsApp mode
- Request: `{ "phone_number": "+1234567890" }`
- POST `/api/whatsapp/stop` - Stop WhatsApp mode

### Slack
- POST `/api/slack` - Send Slack message
- Request: `{ "message": "text", "channel": "#general", "with_tts": true }`
- POST `/api/slack/start` - Start Slack mode
- Request: `{ "channel": "#general" }`
- POST `/api/slack/stop` - Stop Slack mode

## Installation

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" in the top right corner
3. Click "Load unpacked" and select this extension's folder
4. The extension icon should appear in your Chrome toolbar
5. **Pin the extension**: Click the puzzle piece icon in Chrome toolbar and click the pin icon next to TalkiTo

## Usage

### Automatic Monitoring (Popup Interface)

1. Make sure your TalkiTo server is running (ports 8000-8010)
2. Click the TalkiTo extension icon in your toolbar
3. Use the toggles to control monitoring:
   - **TalkiTo Monitor**: Master toggle - automatically enables on bolt.new, v0.dev, replit.com
   - **TTS**: Text-to-Speech output (auto-enabled with main toggle)
   - **WhatsApp**: Send outputs to WhatsApp (prompts for number first time)
   - **Slack**: Send outputs to Slack (prompts for channel first time)
4. Navigate to bolt.new, v0.dev, or replit.com - monitoring starts automatically

### Manual Selection (Context Menu)

1. Right-click on any HTML element with text content
2. Select "TalkiTo" → choose your preferred action:
   - **Talk**: Speaks the text via TTS
   - **WhatsApp**: Sends text via WhatsApp (prompts for phone number on first use)
   - **Slack**: Sends text via Slack (prompts for channel on first use)
3. The element's text content will be sent to your server
4. Any changes to the text content will automatically trigger new requests
5. Use Ctrl+Shift+L to stop monitoring the current element

## First-Time Setup

### WhatsApp
- On first use, you'll be prompted to enter your WhatsApp phone number
- Format: `+1234567890` (include country code)
- This number is saved for future use

### Slack
- On first use, you'll be prompted to enter a Slack channel
- Format: `#general` for channels or `@username` for direct messages
- This channel is saved for future use

## Visual Indicators

The extension highlights monitored elements with colored outlines:
- **Red outline**: Talk mode
- **Green outline**: WhatsApp mode  
- **Purple outline**: Slack mode

## Permissions Required

- `contextMenus`: To add the right-click menu options
- `activeTab`: To access the current tab's content
- `scripting`: To inject scripts and show prompts
- `host_permissions`: To make requests to localhost ports 8000-8010

## Technical Details

The extension:
- **Auto-discovers** server port by testing ports 8000-8010
- Uses **MutationObserver** to detect DOM and text content changes
- Makes **POST requests** to appropriate API endpoints
- Includes **error handling** and fallback to console logging
- **Avoids duplicate requests** by tracking text content changes
- **Stores user preferences** (phone number, Slack channel) in memory

## API Endpoints

### Talk
```bash
POST http://localhost:PORT/api/speak
Content-Type: application/json
{ "text": "Hello world" }
```

### WhatsApp
```bash
POST http://localhost:PORT/api/whatsapp
Content-Type: application/json
{
  "message": "Hello from TalkiTo!",
  "to_number": "+1234567890",
  "with_tts": true
}
```

### Slack
```bash
POST http://localhost:PORT/api/slack
Content-Type: application/json
{
  "message": "Hello team!",
  "channel": "#general",
  "with_tts": true
}
```

## Keyboard Shortcuts

- `Ctrl+Shift+L`: Stop monitoring the current element

## Troubleshooting

If the extension isn't working:
1. Check that it's enabled in `chrome://extensions/`
2. Make sure your server is running on any port 8000-8010 with the required endpoints
3. Refresh the webpage after installing/updating the extension
4. Make sure you're right-clicking on elements that contain text
5. Check the browser console for error messages or server connection issues
6. If server is unavailable, the extension will fall back to console logging
7. For WhatsApp/Slack: Ensure you've entered valid phone number/channel when prompted