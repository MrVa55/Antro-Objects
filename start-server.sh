#!/bin/bash

source .env/bin/activate
python3 whisper_online_server.py  --host localhost  --port 4500 --model medium.en --vad
