#!/bin/bash
# TalkiTo Uninstaller

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🗑️  Uninstalling TalkiTo..."
echo ""

# Check if talkito is installed
if [ ! -f "$HOME/.local/bin/talkito" ]; then
    echo -e "${YELLOW}⚠️  TalkiTo doesn't appear to be installed via the quick installer${NC}"
    echo "   No launcher found at ~/.local/bin/talkito"
    exit 1
fi

# Get installation path
if [ -f "$HOME/.config/talkito/install_path" ]; then
    INSTALL_PATH=$(cat "$HOME/.config/talkito/install_path")
    if [ -d "$INSTALL_PATH" ]; then
        echo "📁 Found installation at: $INSTALL_PATH"
        echo -n "   Removing installation directory... "
        rm -rf "$INSTALL_PATH"
        echo -e "${GREEN}✓${NC}"
    fi
fi

# Remove launcher script
echo -n "🔗 Removing launcher script... "
rm -f "$HOME/.local/bin/talkito"
echo -e "${GREEN}✓${NC}"

# Remove config directory
if [ -d "$HOME/.config/talkito" ]; then
    echo -n "⚙️  Removing configuration directory... "
    rm -rf "$HOME/.config/talkito"
    echo -e "${GREEN}✓${NC}"
fi

echo ""
echo -e "${GREEN}✅ TalkiTo has been uninstalled successfully!${NC}"
echo ""
echo "Note: System dependencies (portaudio, etc.) were not removed"
echo "as they might be used by other applications."