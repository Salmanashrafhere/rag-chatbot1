
import os
import time
import tempfile

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from rag_utils import build_or_load_index, update_index_with_files, retrieve

st.set_page_config(page_title="RAG Chatbot (Free)", page_icon="💬")

st.title("💬 RAG Chatbot (Free • Streamlit • Local)")
st.caption("ChromaDB + Sentence-Transformers + FLAN-T5")

# Sidebar controls
st.sidebar.header("Settings & Indexing")
data_dir = st.sidebar.text_input("Data folder", value="data")
persist_dir = st.sidebar.text_input("ChromaDB path", value="chroma_db")
top_k = st.sidebar.slider("Top-k retrieved chunks", 1, 8, 3)

default_model = os.getenv("FLAN_T5_MODEL", "google/flan-t5-base")
model_name = st.sidebar.text_input("HF Model (seq2seq)", value=default_model)
build_index = st.sidebar.button("🔧 Build/Load Index from Data Folder")
uploaded_files = st.sidebar.file_uploader("Upload PDFs/TXT/MD to add", type=["pdf","txt","md"], accept_multiple_files=True)
add_to_index = st.sidebar.button("➕ Add Uploaded to Index")
clear_history = st.sidebar.button("🧹 Clear Chat History")

if "collection" not in st.session_state:
    st.session_state.collection = None
if "emb_model" not in st.session_state:
    st.session_state.emb_model = None
if "pipe" not in st.session_state:
    st.session_state.pipe = None
if "history" not in st.session_state:
    st.session_state.history = []

# Load / build index
if build_index:
    with st.spinner("Building / loading index..."):
        col, emb_model = build_or_load_index(data_dir=data_dir, persist_dir=persist_dir)
        st.session_state.collection = col
        st.session_state.emb_model = emb_model
    st.success("Index ready!")

# Add uploaded files
if add_to_index and uploaded_files:
    temp_paths = []
    for uf in uploaded_files:
        suffix = os.path.splitext(uf.name)[1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uf.read())
        tmp.flush()
        temp_paths.append(tmp.name)
    with st.spinner("Adding files to index..."):
        added = update_index_with_files(temp_paths, persist_dir=persist_dir)
    st.success(f"Added {added} chunks from {len(uploaded_files)} files to index.")

# Clear history
if clear_history:
    st.session_state.history = []
    st.experimental_rerun()

# Load generation model
def get_pipe():
    if st.session_state.pipe is None or st.session_state.pipe.model.name_or_path != model_name:
        with st.spinner(f"Loading generation model: {model_name} (first time may take a while)..."):
            tok = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            st.session_state.pipe = pipeline("text2text-generation", model=model, tokenizer=tok)
    return st.session_state.pipe

# Chat UI
st.subheader("Chat")
user_q = st.text_input("Ask a question about your documents", key="user_q")

if st.button("Ask") and user_q.strip():
    if st.session_state.collection is None or st.session_state.emb_model is None:
        st.error("Please build/load the index first from the sidebar.")
    else:
        # Retrieve
        result = retrieve(user_q, st.session_state.collection, st.session_state.emb_model, top_k=top_k)
        contexts = result["documents"]
        metas = result["metadatas"]
        dists = result["distances"]
        # Build prompt
        context_block = ""
        for i, (c, m) in enumerate(zip(contexts, metas)):
            src = m.get("source", "unknown")
            context_block += f"[{i+1}] Source: {src}\n{c}\n\n"
        prompt = f"""You are a helpful assistant. Answer the QUESTION using only the CONTEXT.
If you are unsure, say you don't know.

CONTEXT:
{context_block}

QUESTION: {user_q}
ANSWER:
"""
        pipe = get_pipe()
        gen = pipe(prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"].strip()

        # Save and display
        st.session_state.history.append(("user", user_q))
        st.session_state.history.append(("bot", gen))

        # Show answer and sources
        st.markdown("### Answer")
        st.write(gen)
        with st.expander("Show retrieved sources"):
            for i, (c, m, dist) in enumerate(zip(contexts, metas, dists)):
                st.markdown(f"**{i+1}) {m.get('source','unknown')}**  \nDistance: {dist:.4f}")
                st.code(c)

# History display
if st.session_state.history:
    st.markdown("---")
    st.markdown("### Chat History")
    for role, msg in st.session_state.history:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Bot:** {msg}")
