#!/bin/bash
# TalkiTo Quick Installer
# This script provides a one-liner installation method for talkito

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect OS
OS="$(uname -s)"
ARCH="$(uname -m)"

echo "🎙️  Installing TalkiTo..."
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install system dependencies based on OS
install_system_deps() {
    case "$OS" in
        Darwin*)
            echo "📦 Installing macOS dependencies..."
            if ! command_exists brew; then
                echo -e "${RED}❌ Homebrew not found. Please install from https://brew.sh${NC}"
                exit 1
            fi
            
            # Install portaudio if not present
            if ! brew list portaudio &>/dev/null; then
                echo "   Installing portaudio..."
                brew install portaudio
            else
                echo "   ✓ portaudio already installed"
            fi
            ;;
            
        Linux*)
            echo "📦 Installing Linux dependencies..."
            if command_exists apt-get; then
                echo "   Detected Debian/Ubuntu system"
                sudo apt-get update -qq
                sudo apt-get install -y python3-pip python3-venv portaudio19-dev git
            elif command_exists yum; then
                echo "   Detected Red Hat/Fedora system"
                sudo yum install -y python3-pip python3-venv portaudio-devel git
            elif command_exists pacman; then
                echo "   Detected Arch Linux system"
                sudo pacman -Sy --noconfirm python-pip python-virtualenv portaudio git
            elif command_exists apk; then
                echo "   Detected Alpine Linux system"
                sudo apk add --no-cache python3 py3-pip portaudio-dev git
            else
                echo -e "${YELLOW}⚠️  Could not detect package manager. Please install these manually:${NC}"
                echo "   - Python 3.10+"
                echo "   - pip3"
                echo "   - portaudio development files"
                echo "   - git"
                exit 1
            fi
            ;;
            
        *)
            echo -e "${RED}❌ Unsupported OS: $OS${NC}"
            echo "   talkito currently supports macOS and Linux"
            exit 1
            ;;
    esac
}

# Create virtual environment and install talkito
install_talkito() {
    echo ""
    echo "🚀 Installing talkito..."
    
    # Create a temporary directory for installation
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"
    
    # Clone the repository
    echo "   Cloning repository..."
    git clone --quiet https://github.com/robdmac/talkito.git
    cd talkito
    
    # Create virtual environment
    echo "   Creating virtual environment..."
    python3 -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    echo "   Upgrading pip..."
    pip install --quiet --upgrade pip
    
    # Install talkito (all features included by default)
    echo "   Installing talkito..."
    pip install --quiet -e .
    
    # Create a wrapper script
    echo "   Creating launcher script..."
    INSTALL_DIR="$HOME/.local/bin"
    mkdir -p "$INSTALL_DIR"
    
    cat > "$INSTALL_DIR/talkito" << EOF
#!/bin/bash
# TalkiTo launcher script
"$TEMP_DIR/talkito/venv/bin/python" "$TEMP_DIR/talkito/talkito.py" "\$@"
EOF
    
    chmod +x "$INSTALL_DIR/talkito"
    
    # Store installation path for potential uninstall
    mkdir -p "$HOME/.config/talkito"
    echo "$TEMP_DIR/talkito" > "$HOME/.config/talkito/install_path"
    
    echo -e "${GREEN}   ✓ Installation complete${NC}"
}

# Check system requirements
check_requirements() {
    echo "🔍 Checking system requirements..."
    
    # Check Python version
    if ! command_exists python3; then
        echo -e "${RED}❌ Python 3 is required but not found${NC}"
        echo "   Please install Python 3.10 or higher"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    REQUIRED_VERSION="3.10"
    
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
        echo -e "${RED}❌ Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)${NC}"
        exit 1
    fi
    
    echo "   ✓ Python $PYTHON_VERSION"
    
    # Check git
    if ! command_exists git; then
        echo -e "${YELLOW}⚠️  git not found, will install${NC}"
    else
        echo "   ✓ git installed"
    fi
}

# Main installation flow
main() {
    # Print banner
    echo -e "${BLUE}╔═══════════════════════════════════╗${NC}"
    echo -e "${BLUE}║       TalkiTo Quick Installer     ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════╝${NC}"
    echo ""
    
    # Check requirements
    check_requirements
    
    # Install dependencies
    echo ""
    install_system_deps
    
    # Install talkito
    install_talkito
    
    # Update PATH if needed
    export PATH="$HOME/.local/bin:$PATH"
    
    # Success message
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║     ✅ TalkiTo installed successfully!    ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"
    echo ""
    echo "🎯 Quick start:"
    echo -e "   ${BLUE}talkito claude${NC}"
    echo ""
    echo "📖 For more options:"
    echo -e "   ${BLUE}talkito --help${NC}"
    echo ""
    
    # Check if PATH needs to be updated
    if ! echo $PATH | grep -q "$HOME/.local/bin"; then
        echo -e "${YELLOW}⚠️  Important: Add ~/.local/bin to your PATH${NC}"
        echo ""
        echo "Add this line to your shell configuration file:"
        echo ""
        if [[ "$SHELL" == *"zsh"* ]]; then
            echo -e "   ${BLUE}echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.zshrc${NC}"
            echo -e "   ${BLUE}source ~/.zshrc${NC}"
        else
            echo -e "   ${BLUE}echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc${NC}"
            echo -e "   ${BLUE}source ~/.bashrc${NC}"
        fi
        echo ""
        echo "Or run talkito directly with:"
        echo -e "   ${BLUE}~/.local/bin/talkito claude${NC}"
    fi
}

# Handle errors
trap 'echo -e "\n${RED}❌ Installation failed. Please check the error messages above.${NC}"' ERR

# Run main function
main "$@"
