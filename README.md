# 🧠 Local LLM Chat App (with RAG, Vision, and CSV Intelligence)

This is a secure, locally hosted AI assistant built using [Streamlit](https://streamlit.io/) and [Ollama](https://ollama.com/) that supports:

- 📄 Document-based Q&A (PDF, TXT)
- 🖼 Image-based visual question answering (e.g., with `llama3.2-vision`)
- 📊 CSV data analysis and chart generation using LLMs
- 🔁 Supports follow-up questions with session history
- ⚡️ GPU-accelerated via Ollama + SentenceTransformers

---

## 🗂 Project Structure

```
OllamaInstance/
├── app.py                      # Main Streamlit entrypoint
├── config.py                   # Constants and environment settings
├── logger.py                   # Shared logger
├── models.py                   # Model loader + embed model
├── requirements.txt            # Python dependencies
├── .gitignore
├── README.md
└── utils/
    ├── csv_utils.py            # LLM chart generation from CSV
    ├── document_utils.py       # RAG chunking, embedding, search
    ├── vision_utils.py         # Image to LLM interface
```

---

## 🚀 Getting Started

### ✅ 1. Install Ollama and pull models

Install [Ollama](https://ollama.com) and run:

```bash
ollama pull mistral
ollama pull llama3.2-vision
ollama pull deepseek-coder:6.7b
```

Then start Ollama server:
```bash
ollama serve
```

---

### ✅ 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # On Windows
# or
source venv/bin/activate     # On macOS/Linux
```

---

### ✅ 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

### ✅ 4. Run the app

```bash
streamlit run app.py
```

Then open your browser at [http://localhost:8501](http://localhost:8501)

---

## 🔐 Notes

- All data remains local (no external API calls)
- Suitable for internal organizational use or offline demo deployments
- Easily extensible: Add audio transcription, FAISS, LangChain, etc.

---

## 🤝 Contributors

- Built and maintained by **[NICTU]**
- Inspired by Ollama, Streamlit, SentenceTransformers, Matplotlib, FAISS
