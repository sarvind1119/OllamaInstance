# utils/document_utils.py

import bleach
import hashlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from logger import get_logger

logger = get_logger(__name__)

CHUNK_SIZE = 500
OVERLAP = 100

def sanitize_input(text):
    return bleach.clean(text, tags=[], strip=True)

def split_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Split text into overlapping chunks for embedding."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

def embed_chunks(chunks, model):
    """Generate embeddings for all text chunks."""
    return model.encode(chunks, show_progress_bar=False)

def get_top_k_chunks(query, chunks, embeddings, model, k=3):
    """Get the top-k relevant chunks for a given query using cosine similarity."""
    q_embed = model.encode([sanitize_input(query)])[0]
    sims = cosine_similarity([q_embed], embeddings)[0]
    top_k = np.argsort(sims)[-k:][::-1]
    return [chunks[i] for i in top_k]

def process_document(doc_text, session_state):
    """Embed and cache the document if it's new or changed."""
    if not doc_text.strip():
        return False

    text_hash = hashlib.sha256(doc_text.encode()).hexdigest()

    if session_state.doc_cache["text_hash"] != text_hash:
        logger.info("Processing new document and generating embeddings")
        chunks = split_text(doc_text)
        embeddings = embed_chunks(chunks, session_state.embed_model)
        session_state.doc_cache.update({
            "text_hash": text_hash,
            "chunks": chunks,
            "embeddings": embeddings,
            "text": doc_text
        })
    return True
