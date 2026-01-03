FROM python:3.8.5-slim-buster

WORKDIR /app


COPY . /app

# Fix for old Debian Buster repositories and install build tools
RUN sed -i 's|http://deb.debian.org/debian|http://archive.debian.org/debian|g' /etc/apt/sources.list && \
	sed -i 's|http://security.debian.org/debian-security|http://archive.debian.org/debian-security|g' /etc/apt/sources.list && \
	apt-get update && apt-get install -y --no-install-recommends gcc g++ make && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]

