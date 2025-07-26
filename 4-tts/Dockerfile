FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('charactr/vocos-mel-24khz')"

EXPOSE 8003

CMD ["python3","tts_server.py"]
