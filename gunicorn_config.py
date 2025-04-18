# gunicorn_config.py

# Number of worker processes for handling requests
workers = 4  # A good rule is 2-4 x number of CPU cores

# Number of threads per worker
threads = 2

# Socket to bind
bind = "0.0.0.0:$PORT"  # This allows Render to set the port via environment variable

# Timeout (in seconds) for a request to be processed
timeout = 120  # Longer timeout for ML inference

# Maximum number of requests a worker will process before restarting
max_requests = 1000
max_requests_jitter = 50  # Adds randomness to avoid all workers restarting at once

# Process name
proc_name = 'dementia_audio_app'

# Access log format
accesslog = '-'  # Log to stdout
errorlog = '-'   # Log errors to stdout
loglevel = 'info'

# Preload application code before forking worker processes
preload_app = True

# Set environment variables
raw_env = [
    'PYTHONUNBUFFERED=1'  # Ensures that Python output is sent straight to terminal
]
