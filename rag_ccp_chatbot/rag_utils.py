
import os
import math
import uuid
from typing import List, Dict, Tuple

import numpy as np
import chromadb
from chromadb import Client
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# Embedding model (small, fast, free)
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

def load_documents_from_dir(data_dir: str) -> List[Dict]:
    """Load .txt, .md, .pdf from a directory tree."""
    docs = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            path = os.path.join(root, fn)
            if fn.lower().endswith(".txt") or fn.lower().endswith(".md"):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                docs.append({"source": path, "text": text})
            elif fn.lower().endswith(".pdf"):
                try:
                    reader = PdfReader(path)
                    pages = [p.extract_text() or "" for p in reader.pages]
                    text = "\n".join(pages)
                    docs.append({"source": path, "text": text})
                except Exception as e:
                    print(f"[WARN] Could not parse PDF: {path}: {e}")
    return docs

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Naive chunking by characters with overlap."""
    chunks = []
    n = len(text)
    i = 0
    while i < n:
        end = min(i + chunk_size, n)
        chunk = text[i:end]
        chunks.append(chunk)
        if end == n:
            break
        i = end - overlap
        if i < 0 or i >= n:
            break
    return [c.strip() for c in chunks if c.strip()]

def build_or_load_index(
    data_dir: str,
    persist_dir: str = "chroma_db",
    collection_name: str = "docs"
) -> Tuple[chromadb.api.models.Collection.Collection, SentenceTransformer]:
    """Build Chroma index if empty; otherwise load existing. Returns (collection, emb_model)."""
    os.makedirs(persist_dir, exist_ok=True)
    client = Client()

    try:
        col = client.get_collection(collection_name)
    except Exception:
        col = client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})
    emb_model = SentenceTransformer(EMB_MODEL_NAME)

    # If collection is empty, (re)index from data_dir
    if col.count() == 0:
        docs = load_documents_from_dir(data_dir)
        if not docs:
            print(f"[INFO] No documents found in {data_dir}.")
            return col, emb_model

        ids, texts, metas, embs = [], [], [], []
        for d in docs:
            chunks = chunk_text(d["text"])
            for idx, ch in enumerate(chunks):
                uid = str(uuid.uuid4())
                ids.append(uid)
                texts.append(ch)
                metas.append({"source": d["source"], "chunk_id": idx})
        if texts:
            vecs = emb_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            embs = [v.tolist() for v in vecs]
            col.add(ids=ids, documents=texts, metadatas=metas, embeddings=embs)
            print(f"[OK] Indexed {len(texts)} chunks from {len(docs)} documents.")
    return col, emb_model

def update_index_with_files(
    file_paths: List[str],
    persist_dir: str = "chroma_db",
    collection_name: str = "docs"
) -> int:
    """Add/append new files to an existing (or fresh) Chroma index."""
    os.makedirs(persist_dir, exist_ok=True)
    client = PersistentClient(path=persist_dir)
    try:
        col = client.get_collection(collection_name)
    except Exception:
        col = client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})
    emb_model = SentenceTransformer(EMB_MODEL_NAME)

    added = 0
    for path in file_paths:
        text = ""
        if path.lower().endswith((".txt", ".md")):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif path.lower().endswith(".pdf"):
            try:
                reader = PdfReader(path)
                pages = [p.extract_text() or "" for p in reader.pages]
                text = "\n".join(pages)
            except Exception as e:
                print(f"[WARN] Could not parse PDF: {path}: {e}")
        else:
            continue

        chunks = chunk_text(text)
        if not chunks:
            continue

        texts = []
        metas = []
        ids = []
        for idx, ch in enumerate(chunks):
            texts.append(ch)
            metas.append({"source": path, "chunk_id": idx})
            ids.append(str(uuid.uuid4()))
        vecs = emb_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        col.add(ids=ids, documents=texts, metadatas=metas, embeddings=[v.tolist() for v in vecs])
        added += len(texts)
    return added

def retrieve(
    question: str,
    col,
    emb_model,
    top_k: int = 3
) -> Dict:
    """Query the vector DB with the question; return docs, metadatas, distances."""
    qvec = emb_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()
    out = col.query(query_embeddings=[qvec], n_results=top_k, include=["distances", "documents", "metadatas"])
    # Normalize result shape
    result = {
        "documents": out.get("documents", [[]])[0],
        "metadatas": out.get("metadatas", [[]])[0],
        "distances": out.get("distances", [[]])[0],
    }
    return result
