#!/bin/bash

# Upgrade huggingface_hub
pip install --upgrade huggingface_hub

# Download the model locally
hf download Supertone/supertonic-3 --local-dir supertonic3 --quiet

# Install requirements.txt
pip install -q -r requirements.txt

echo "Setup finished successfully!"