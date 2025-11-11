#!/bin/bash
# Railway startup script - properly handles $PORT environment variable

if [ -z "$PORT" ]; then
  echo "PORT not set, using default 5000"
  PORT=5000
fi

echo "Starting gunicorn on port $PORT"
exec gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 2
