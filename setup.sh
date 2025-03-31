#!/bin/bash

echo "Setting up the AI Chatbot..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
echo "Setup complete! Run the chatbot with: source venv/bin/activate && python3.11 oop_gui_chatbot.py"
