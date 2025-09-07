# RAG Chatbot (Free, Streamlit GUI)

This is a **Retrieval-Augmented Generation (RAG)** chatbot template designed for student CCP projects.
It uses **ChromaDB** for local vector search (free) and a **local Hugging Face model (FLAN-T5)** for generation (free).
A simple **Streamlit GUI** is included that shows answers *and* retrieved source snippets.

---

## Features
- **Free** stack: Sentence-Transformers embeddings + ChromaDB (local) + FLAN-T5 (local)
- **Streamlit GUI** with chat history
- Shows retrieved chunks + source document names
- Works **offline** after first-time model download
- Easy deploy to **Streamlit Cloud** or **Hugging Face Spaces**

---

## Folder Structure
```
rag_ccp_chatbot/
├─ app.py
├─ rag_utils.py
├─ requirements.txt
├─ data/
│  └─ sample_docs/
│     ├─ intro_nlp.txt
│     └─ course_policy.txt
└─ chroma_db/   (created automatically after first indexing)
```

---

## 1) Local Setup (Windows/Mac/Linux)

```bash
# create env (recommended)
python -m venv .venv
# activate
# Windows:
.venv\Scripts\activate
# Mac/Linux:
# source .venv/bin/activate

pip install -r requirements.txt

# run app
streamlit run app.py
```

Then open the URL shown by Streamlit (usually http://localhost:8501).

> First run will download a small Hugging Face model (FLAN-T5) automatically.
> After that you can run **offline** for retrieval + generation.

---

## 2) How to Use
1. Place your PDFs/TXT/MD files in `data/` (or upload via the Streamlit sidebar).
2. Click **Build/Update Index** to create a Chroma vector index.
3. Ask a question in the chat box. The app will:
   - retrieve top-k chunks
   - generate an answer with the selected model
   - show sources and distances

---

## 3) Deploy Options (Free)

### A) Streamlit Cloud
1. Push this folder to a **public GitHub repo**.
2. Go to Streamlit Cloud → New app → connect your repo.
3. **Main file**: `app.py`
4. **Python version**: 3.10+ (recommended)
5. Add `requirements.txt` (already provided).

### B) Hugging Face Spaces (Streamlit SDK)
1. Create a new Space → **SDK: Streamlit** → **Hardware: CPU Basic**.
2. Upload `app.py`, `rag_utils.py`, `requirements.txt`, and the `data/` folder (optional).
3. The app will auto-build and run.
4. For updates, push to the Space or connect to your GitHub.

> Tip: Keep your model as `google/flan-t5-base` (default). If hardware is weak, set `FLAN_T5_MODEL=google/flan-t5-small` in Space secrets/variables and the app will use the smaller model automatically.

### C) Kaggle Notebook
- Upload this project as a zip or individual files to Kaggle.
- You can **test** indexing & Q/A in a notebook (install the same `requirements.txt`).
- For a GUI demo on Kaggle, prefer creating a **Hugging Face Space** or **Streamlit Cloud** and share that public link in your CCP.

---

## 4) Tech Notes
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (fast & small).
- **Vector DB**: Chroma (persistent at `./chroma_db`).
- **LLM**: `google/flan-t5-base` by default (change via env var `FLAN_T5_MODEL`), fully local after first download.
- **PDF parsing**: `pypdf`

---

## 5) RAG Architecture (High Level)
User Query → Embed → Retrieve top-k chunks (Chroma) →
Build prompt with context → Generate answer (FLAN-T5) →
Show answer + sources.

---

## 6) License
Educational use. Customize freely for your course CCP.
