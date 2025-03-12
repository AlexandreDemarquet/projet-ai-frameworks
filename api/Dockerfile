FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app
COPY . /app

RUN apt update && apt install -y python3 python3-pip && \
    pip3 install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]

