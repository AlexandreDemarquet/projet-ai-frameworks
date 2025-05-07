FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

COPY requirements.txt /app

RUN apt-get update && \
    apt-get install -y python3 python3-pip curl && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install gdown

COPY . /app

RUN mkdir -p /weights

RUN gdown "https://drive.google.com/uc?export=download&id=1rJ8TqbqQvk7uQnmOh9mJbk_m4anlh8f-" -O /weights/filmClassifier.pth

EXPOSE 5000

CMD ["python3", "app.py"]

