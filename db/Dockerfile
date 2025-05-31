FROM python:3.10

WORKDIR /app

COPY requirements.txt /app

RUN apt update && apt install -y python3 python3-pip && \
    pip3 install --no-cache-dir -r requirements.txt

# RUN pip install gdown

COPY . /app

#RUN python /app/download_data.py
RUN python download_data.py && python buildIndexDb.py

EXPOSE 8000

CMD ["gunicorn", "-b", "0.0.0.0:8000", "api_db:app"]
