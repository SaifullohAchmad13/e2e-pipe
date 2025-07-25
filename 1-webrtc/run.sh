#!/bin/bash

start_servers() {
  source .env 2>/dev/null

  python3 server.py -t webrtc > log_webrtc.log 2>&1 &
  WEBRTC_PID=$!
  echo $WEBRTC_PID > webrtc.pid
}

stop_servers() {
  if [ -f webrtc.pid ]; then
    WEBRTC_PID=$(cat webrtc.pid)
    echo "Stopping WebRTC server (PID: $WEBRTC_PID)..."
    kill $WEBRTC_PID
    rm webrtc.pid
  else
    echo "No WebRTC server PID file found."
  fi
}

# Main script execution
if [ "$1" == "start" ]; then
  echo "Starting WebRTC server..."
  start_servers
elif [ "$1" == "stop" ]; then
  echo "Stopping WebRTC server..."
  stop_servers
else
  echo "Usage: $0 [start|stop]"
  exit 1
fi
