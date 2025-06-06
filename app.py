import streamlit as st
import requests
import json
import subprocess
import os
import logging
from io import StringIO
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import bleach
import hashlib
import base64
from PIL import Image
import io
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Local LLM Chat", layout="wide")

OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
CHUNK_SIZE = 500
OVERLAP = 100
OLLAMA_TIMEOUT = 180  # seconds

with st.sidebar:
    st.image("https://www.lbsnaa.gov.in/admin_assets/images/logo.png", width=250)
    st.header("About App")
    st.markdown("""
    Secure, local LLM chat for organizations. Supports document-based RAG and vision queries. Run entirely on your local machine.
    """)

def sanitize_input(text):
    return bleach.clean(text, tags=[], strip=True)

def split_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

def embed_chunks(chunks, model):
    return model.encode(chunks, show_progress_bar=False)

def get_top_k_chunks(query, chunks, embeddings, model, k=3):
    q_embed = model.encode([sanitize_input(query)])[0]
    sims = cosine_similarity([q_embed], embeddings)[0]
    top_k = np.argsort(sims)[-k:][::-1]
    return [chunks[i] for i in top_k]

def process_document(doc_text):
    if not doc_text.strip():
        st.error("The document is empty or could not be processed.")
        return False
    text_hash = hashlib.sha256(doc_text.encode()).hexdigest()
    if st.session_state.doc_cache["text_hash"] != text_hash:
        with st.spinner("Processing document..."):
            logger.info("Generating embeddings for new document")
            chunks = split_text(doc_text)
            embeddings = embed_chunks(chunks, st.session_state.embed_model)
            st.session_state.doc_cache.update({
                "text_hash": text_hash,
                "chunks": chunks,
                "embeddings": embeddings,
                "text": doc_text
            })
    return True

def call_llama_vision(image_bytes, prompt, ollama_url="http://localhost:11434", model="llama3.2-vision"):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.thumbnail((512, 512))
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        encoded_img = base64.b64encode(buf.getvalue()).decode("utf-8")

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [encoded_img]
                }
            ],
            "stream": False
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{ollama_url}/api/chat", json=payload, headers=headers, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()

        response_data = response.json()
        return response_data.get("message", {}).get("content", "[No content returned]")

    except requests.exceptions.RequestException as e:
        logger.error(f"Vision query failed: {e}")
        return f"❌ Vision Request Failed: {e}"

    except Exception as e:
        logger.error(f"Unexpected error in vision function: {e}")
        return f"❌ Unexpected Vision Error: {e}"

@st.cache_resource
def load_embed_model():
    logger.info("Loading SentenceTransformer model")
    return SentenceTransformer('all-MiniLM-L6-v2')

if "embed_model" not in st.session_state:
    st.session_state.embed_model = load_embed_model()

if "doc_cache" not in st.session_state:
    st.session_state.doc_cache = {"text_hash": None, "chunks": None, "embeddings": None, "text": None}

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_image_bytes" not in st.session_state:
    st.session_state.last_image_bytes = None

def get_ollama_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        models = [line.split()[0] for line in result.stdout.splitlines()[1:] if line]
        return sorted(models)
    except Exception as e:
        logger.error(f"Failed to fetch models: {e}")
        return ["mistral"]

models = get_ollama_models()
model = st.selectbox("Choose a model", models, index=0)

csv_mode = st.checkbox("Ask questions about a CSV file")

if csv_mode:
    uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.write("📊 Data Preview", df.head())

        csv_prompt = st.chat_input("Ask something about this data")
        if csv_prompt:
            st.chat_message("user").markdown(csv_prompt)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing CSV..."):
                    try:
                        prompt = f"Given the following dataframe:\n{df.head(100).to_csv(index=False)}\n\nGenerate a matplotlib or pandas chart based on this request: {csv_prompt}"

                        response = requests.post(
                            f"{OLLAMA_ENDPOINT}/api/generate",
                            json={
                                "model": model,
                                "prompt": prompt,
                                "stream": False
                            },
                            timeout=OLLAMA_TIMEOUT
                        )
                        response.raise_for_status()
                        result = response.json()["response"]

                        if "plot" in result or "plt." in result:
                            # Safe wrapper for dynamic code execution
                            st.info("🔄 Executing generated chart...")
                            with st.empty():
                                fig = plt.figure()
                                exec_globals = {"df": df, "pd": pd, "plt": plt, "fig": fig}
                                exec(result, exec_globals)
                                st.pyplot(fig)
                        else:
                            st.markdown(result)

                    except Exception as e:
                        logger.error(f"CSV chart error: {e}")
                        st.error("Failed to generate or render chart.")
# File upload and document processing

doc_mode = st.checkbox("Ask from your document")
doc_text = ""
if doc_mode:
    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    if uploaded_file and uploaded_file.size <= MAX_FILE_SIZE:
        try:
            if uploaded_file.type == "application/pdf":
                with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                    doc_text = "\n".join([page.get_text() for page in doc])
            elif uploaded_file.type == "text/plain":
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                doc_text = stringio.read()
            if doc_text:
                process_document(doc_text)
        except Exception as e:
            st.error(f"Failed to process document: {e}")

vision_mode = st.checkbox("Ask questions about an image")
image_prompt = ""
if vision_mode:
    uploaded_image = st.file_uploader("Upload image (JPG or PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image_bytes = uploaded_image.read()
        st.session_state.last_image_bytes = image_bytes

    if st.session_state.last_image_bytes:
        st.image(st.session_state.last_image_bytes, caption="Image in context", use_column_width=True)
        image_prompt = st.chat_input("Ask a question about this image")

if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.last_image_bytes = None
    st.experimental_rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if vision_mode and st.session_state.last_image_bytes and image_prompt:
    st.session_state.messages.append({"role": "user", "content": f"[Image] {image_prompt}"})
    st.chat_message("user").markdown(f"🖼️ {image_prompt}")

    with st.chat_message("assistant"):
        with st.spinner("Analyzing image..."):
            result = call_llama_vision(st.session_state.last_image_bytes, image_prompt, model=model)
            st.markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})

user_input = st.chat_input("Ask something...")
if user_input:
    sanitized_input = sanitize_input(user_input)
    st.session_state.messages.append({"role": "user", "content": sanitized_input})
    st.chat_message("user").markdown(sanitized_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            final_prompt = sanitized_input
            if doc_mode and st.session_state.doc_cache["text"]:
                top_chunks = get_top_k_chunks(
                    sanitized_input,
                    st.session_state.doc_cache["chunks"],
                    st.session_state.doc_cache["embeddings"],
                    st.session_state.embed_model,
                    k=3
                )
                context = "\n\n".join(top_chunks)
                final_prompt = f"Answer based only on the below document context:\n\n{context}\n\nQuestion: {sanitized_input}"

            try:
                response = requests.post(
                    f"{OLLAMA_ENDPOINT}/api/generate",
                    json={
                        "model": model,
                        "prompt": final_prompt,
                        "stream": True
                    },
                    stream=True,
                    timeout=30
                )
                response.raise_for_status()

                full_response = ""
                for line in response.iter_lines():
                    if line:
                        if line.startswith(b"data: "):
                            line = line[6:]
                        data = json.loads(line.decode("utf-8"))
                        token = data.get("response", "")
                        full_response += token

                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except requests.exceptions.RequestException as e:
                st.error(f"Failed to get response: {e}")
