# models.py

import subprocess
from sentence_transformers import SentenceTransformer
from logger import get_logger
import streamlit as st

logger = get_logger(__name__)

@st.cache_resource
def load_embed_model():
    """Load and cache the embedding model."""
    logger.info("Loading SentenceTransformer model")
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_ollama_models():
    """Return a list of available Ollama models installed locally."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        models = [line.split()[0] for line in result.stdout.splitlines()[1:] if line]
        return sorted(models)
    except Exception as e:
        logger.error(f"Failed to fetch models: {e}")
        return ["mistral"]
