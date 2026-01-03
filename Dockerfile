FROM python:3.8.5-slim-buster

WORKDIR /app


COPY . /app

# Install build tools required for lightgbm and other packages
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ make && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]

 