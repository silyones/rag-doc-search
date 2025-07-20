# rag-doc-search: Semantic Document Search with FAISS + Gradio

A lightweight Retrieval-Augmented Generation (RAG) application to semantically search across custom documents. This app uses [FAISS](https://github.com/facebookresearch/faiss) for vector indexing and [SentenceTransformers](https://www.sbert.net/) for embeddings, all wrapped in a clean [Gradio](https://www.gradio.app/) user interface.

> Ask a question and retrieve the most relevant paragraph from your PDF-derived text data.

---

## Features

- Semantic search over document paragraphs
- Supports `.txt` files extracted from PDFs
- Fast and accurate nearest neighbor search using FAISS
- Modern, styled Gradio interface
- Displays source file name and similarity score

---

## Project Structure
rag-doc-search/
├── .gradio/              # Gradio logs/flags (auto-generated)
│   └── flagged/          
├── .venv/                # Virtual environment (optional)
├── data/                 # Folder with .txt files (from PDFs)
├── app.py                # Main app logic (RAG class + Gradio UI)
├── faiss_index.index     # (Optional) Saved FAISS index (auto-generated)
└── requirements.txt      # Python dependencies


## Installation

### 1. Clone the Repository

git clone https://github.com/your-username/mini-rag-doc-search.git
cd rag-doc-search

- if you want to install libraries from requirements
pip install -r requirements.txt

### 2. Run the App
python app.py

## check out the demo of the UI
- demo.png

## Example Questions 
- What does DDoS mitigation involve?
- Zero Trust Principles
- Benefits of Network Segmentation
- What is cybersecurity
