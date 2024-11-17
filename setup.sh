#!/bin/bash

# Function to check the operating system
check_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$NAME
    else
        OS=$(uname -s)
    fi
}

# Function to install dependencies based on the OS
install_dependencies() {
    case $OS in
        "Ubuntu" | "Debian GNU/Linux")
            echo "Installing dependencies for Ubuntu/Debian..."
            sudo apt-get update
            sudo apt-get install -y portaudio19-dev python3-dev tesseract-ocr
            ;;
        "Fedora")
            echo "Installing dependencies for Fedora..."
            sudo dnf install -y portaudio-devel python3-devel tesseract
            ;;
        "Darwin")
            echo "Installing dependencies for macOS..."
            brew install portaudio tesseract
            ;;
        *)
            echo "Unsupported operating system: $OS"
            echo "Please install PortAudio and Tesseract manually."
            exit 1
            ;;
    esac
}

# Function to create and activate virtual environment
setup_python_env() {
    echo "Setting up Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
}

# Main execution
check_os
install_dependencies
setup_python_env

echo "Setup completed successfully!"
echo "To activate the virtual environment, run: source venv/bin/activate"