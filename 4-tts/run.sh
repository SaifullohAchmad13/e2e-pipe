#!/bin/bash

start_servers() {
    source .env 2>/dev/null

    if ! command -v ffmpeg &> /dev/null; then
        echo "Installing ffmpeg..."
        sudo apt install ffmpeg -y
    fi

    python3 tts_server.py > log_tts.log 2>&1 &
    TTS_PID=$!
    echo $TTS_PID > tts.pid
}

stop_servers() {
  if [ -f tts.pid ]; then
    TTS_PID=$(cat tts.pid)
    echo "Stopping TTS server (PID: $TTS_PID)..."
    kill $TTS_PID
    rm tts.pid
  else
    echo "No TTS server PID file found."
  fi
}

if [ "$1" == "start" ]; then
  echo "Starting TTS server..."
  start_servers
elif [ "$1" == "stop" ]; then
  echo "Stopping TTS server..."
  stop_servers
else
  echo "Usage: $0 [start|stop]"
  exit 1
fi
