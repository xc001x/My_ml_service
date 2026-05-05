#!/bin/bash

echo "Start backend server"
cd /app/backend/server

echo "Applying database migrations..."
python manage.py migrate

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Starting Gunicorn server..."
gunicorn server.wsgi --bind 0.0.0.0:8000 --workers 4 --threads 4
