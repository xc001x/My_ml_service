FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn==20.1.0

ADD ./backend /app/backend
ADD ./docker /app/docker
ADD ./research /app/research

RUN mkdir -p /app/backend/server/static
RUN chmod +x /app/docker/backend/wsgi-entrypoint.sh
