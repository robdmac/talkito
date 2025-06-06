#!/bin/bash
# Talkito shell wrapper
# This allows running 'talkito' instead of './talkito.py'

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the directory containing the package
cd "$DIR"

# Execute the talkito module with all arguments
exec python3 -m talkito.cli "$@"