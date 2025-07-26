FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y git && \
    apt-get install -y wget && \
    apt-get install -y libglib2.0-0 && \
    apt-get install -y libsm6 && \
    apt-get install -y libxrender1 && \
    apt-get install -y libxext6 && \
    apt-get install -y libgl1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .


EXPOSE 8005

CMD ["python", "server.py"]
