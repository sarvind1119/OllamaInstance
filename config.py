# config.py

import os

OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
CHUNK_SIZE = 500
OVERLAP = 100
OLLAMA_TIMEOUT = 180  # seconds
LOGO_PATH = "static/logo.png"
