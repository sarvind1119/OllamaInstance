# import streamlit as st
# import pandas as pd
# from config import OLLAMA_ENDPOINT
# from models import load_embed_model, get_ollama_models
# from logger import get_logger
# from utils.document_utils import process_document, get_top_k_chunks, sanitize_input
# from utils.vision_utils import call_llama_vision
# from utils.csv_utils import ask_csv_question, run_chart_code

# # ─────────────────────────────────────────────────────────────
# # Initialize
# logger = get_logger(__name__)
# st.set_page_config(page_title="Local LLM Chat", layout="wide")

# if "embed_model" not in st.session_state:
#     st.session_state.embed_model = load_embed_model()

# if "doc_cache" not in st.session_state:
#     st.session_state.doc_cache = {"text_hash": None, "chunks": None, "embeddings": None, "text": None}

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if "last_image_bytes" not in st.session_state:
#     st.session_state.last_image_bytes = None

# # ─────────────────────────────────────────────────────────────
# # Sidebar
# with st.sidebar:
#     st.image("https://www.lbsnaa.gov.in/admin_assets/images/logo.png", width=250)
#     st.header("About App")
#     st.markdown("""
#     Secure, local LLM chat for organizations. Supports document-based RAG, vision queries, and CSV-based insights.
#     """)

# # ─────────────────────────────────────────────────────────────
# # Model selection
# models = get_ollama_models()
# model = st.selectbox("Choose a model", models, index=0)

# # ─────────────────────────────────────────────────────────────
# # Document RAG section
# doc_mode = st.checkbox("Ask from your document")
# doc_text = ""
# if doc_mode:
#     uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
#     if uploaded_file:
#         if uploaded_file.type == "application/pdf":
#             import fitz
#             with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
#                 doc_text = "\n".join([page.get_text() for page in doc])
#         elif uploaded_file.type == "text/plain":
#             doc_text = uploaded_file.getvalue().decode("utf-8")
#         if doc_text:
#             process_document(doc_text, st.session_state)

# # ─────────────────────────────────────────────────────────────
# # Vision section
# vision_mode = st.checkbox("Ask questions about an image")
# image_prompt = ""
# if vision_mode:
#     uploaded_image = st.file_uploader("Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"])
#     if uploaded_image:
#         st.session_state.last_image_bytes = uploaded_image.read()

#     if st.session_state.last_image_bytes:
#         st.image(st.session_state.last_image_bytes, caption="Current image", use_column_width=True)
#         image_prompt = st.chat_input("Ask a question about this image")

# # ─────────────────────────────────────────────────────────────
# # CSV section
# csv_mode = st.checkbox("Ask questions about a CSV file")
# if csv_mode:
#     uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"])
#     if uploaded_csv:
#         df = pd.read_csv(uploaded_csv)
#         st.write("📊 Data Preview", df.head())

#         csv_prompt = st.chat_input("Ask something about this data")
#         if csv_prompt:
#             st.chat_message("user").markdown(csv_prompt)
#             with st.chat_message("assistant"):
#                 with st.spinner("Analyzing CSV..."):
#                     code = ask_csv_question(df, csv_prompt, model=model)
#                     with st.expander("🧾 Generated Code", expanded=False):
#                         st.code(code, language="python")
#                     if "plt" in code or "plot" in code:
#                         try:
#                             fig = run_chart_code(code, df)
#                             st.pyplot(fig)
#                         except Exception as e:
#                             st.error(str(e))
#                     else:
#                         st.markdown(code)

# # ─────────────────────────────────────────────────────────────
# # Clear chat
# if st.button("Clear Chat History"):
#     st.session_state.messages = []
#     st.session_state.last_image_bytes = None
#     st.experimental_rerun()

# # ─────────────────────────────────────────────────────────────
# # Display history
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# # ─────────────────────────────────────────────────────────────
# # Vision chat handling
# if vision_mode and st.session_state.last_image_bytes and image_prompt:
#     st.session_state.messages.append({"role": "user", "content": f"[Image] {image_prompt}"})
#     st.chat_message("user").markdown(f"🖼️ {image_prompt}")
#     with st.chat_message("assistant"):
#         with st.spinner("Analyzing image..."):
#             result = call_llama_vision(st.session_state.last_image_bytes, image_prompt, model=model)
#             st.markdown(result)
#             st.session_state.messages.append({"role": "assistant", "content": result})

# # ─────────────────────────────────────────────────────────────
# # Text chat (default or document-based)
# user_input = st.chat_input("Ask something...")
# if user_input:
#     sanitized_input = sanitize_input(user_input)
#     st.session_state.messages.append({"role": "user", "content": sanitized_input})
#     st.chat_message("user").markdown(sanitized_input)

#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             final_prompt = sanitized_input
#             if doc_mode and st.session_state.doc_cache["text"]:
#                 top_chunks = get_top_k_chunks(
#                     sanitized_input,
#                     st.session_state.doc_cache["chunks"],
#                     st.session_state.doc_cache["embeddings"],
#                     st.session_state.embed_model,
#                     k=3
#                 )
#                 context = "\n\n".join(top_chunks)
#                 final_prompt = f"Answer based only on the below document context:\n\n{context}\n\nQuestion: {sanitized_input}"

#             try:
#                 response = requests.post(
#                     f"{OLLAMA_ENDPOINT}/api/generate",
#                     json={"model": model, "prompt": final_prompt, "stream": True},
#                     stream=True,
#                     timeout=30
#                 )
#                 response.raise_for_status()

#                 full_response = ""
#                 for line in response.iter_lines():
#                     if line:
#                         if line.startswith(b"data: "):
#                             line = line[6:]
#                         data = json.loads(line.decode("utf-8"))
#                         token = data.get("response", "")
#                         full_response += token

#                 st.markdown(full_response)
#                 st.session_state.messages.append({"role": "assistant", "content": full_response})

#             except requests.exceptions.RequestException as e:
#                 st.error(f"❌ Failed to get response: {e}")


import streamlit as st
import pandas as pd
import requests
import json
from config import OLLAMA_ENDPOINT
from models import load_embed_model, get_ollama_models
from logger import get_logger
from utils.document_utils import process_document, get_top_k_chunks, sanitize_input
from utils.vision_utils import call_llama_vision
from utils.csv_utils import ask_csv_question, run_chart_code

# ─────────────────────────────────────────────────────────────
logger = get_logger(__name__)
st.set_page_config(page_title="Local LLM Chat", layout="wide")

if "embed_model" not in st.session_state:
    st.session_state.embed_model = load_embed_model()

if "doc_cache" not in st.session_state:
    st.session_state.doc_cache = {"text_hash": None, "chunks": None, "embeddings": None, "text": None}

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_image_bytes" not in st.session_state:
    st.session_state.last_image_bytes = None

# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://www.lbsnaa.gov.in/admin_assets/images/logo.png", width=250)
    st.header("About App")
    st.markdown("""
    Upload any document (PDF, TXT, CSV, image). The app auto-detects the type and responds intelligently using local models.
    """)

models = get_ollama_models()
model = st.selectbox("Choose a model", models, index=0)

# ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a document (PDF, TXT, CSV, or image)", type=["pdf", "txt", "csv", "jpg", "jpeg", "png"])
df = None
doc_text = ""
image_bytes = None

if uploaded_file:
    file_type = uploaded_file.type
    filename = uploaded_file.name.lower()

    if file_type == "application/pdf" or filename.endswith(".pdf"):
        import fitz
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            doc_text = "\n".join([page.get_text() for page in doc])
        process_document(doc_text, st.session_state)
        st.success("✅ PDF processed")

    elif file_type == "text/plain" or filename.endswith(".txt"):
        doc_text = uploaded_file.getvalue().decode("utf-8")
        process_document(doc_text, st.session_state)
        st.success("✅ Text file processed")

    elif file_type == "text/csv" or filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        st.success("✅ CSV file loaded")
        st.write("📊 Data Preview", df.head())

    elif file_type.startswith("image/") or filename.endswith((".jpg", ".jpeg", ".png")):
        image_bytes = uploaded_file.read()
        st.session_state.last_image_bytes = image_bytes
        st.image(image_bytes, caption="Uploaded image", use_column_width=True)
        st.success("✅ Image ready for vision Q&A")

# ─────────────────────────────────────────────────────────────
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.last_image_bytes = None
    st.experimental_rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a question...")
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if df is not None:
                code = ask_csv_question(df, user_input, model=model)
                with st.expander("🧾 Generated Code", expanded=False):
                    st.code(code)
                try:
                    fig = run_chart_code(code, df)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(str(e))

            elif st.session_state.last_image_bytes:
                result = call_llama_vision(st.session_state.last_image_bytes, user_input, model=model)
                st.markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})

            elif doc_text:
                chunks = get_top_k_chunks(
                    user_input,
                    st.session_state.doc_cache["chunks"],
                    st.session_state.doc_cache["embeddings"],
                    st.session_state.embed_model,
                    k=3
                )
                context = "\n\n".join(chunks)
                final_prompt = f"Answer using this document:\n{context}\n\nQuestion: {user_input}"

                try:
                    response = requests.post(
                        f"{OLLAMA_ENDPOINT}/api/generate",
                        json={"model": model, "prompt": final_prompt, "stream": True},
                        stream=True,
                        timeout=30
                    )
                    response.raise_for_status()

                    full_response = ""
                    for line in response.iter_lines():
                        if line and line.startswith(b"data: "):
                            token = json.loads(line[6:].decode()).get("response", "")
                            full_response += token

                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Failed to get response: {e}")
            else:
                st.warning("Please upload a document first.")
