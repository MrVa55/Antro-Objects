#!/bin/bash
source .venv/bin/activate
python3 whisper_online_server.py --warmup-file jfk.wav --host localhost --port 4500 --model medium.en --vad

