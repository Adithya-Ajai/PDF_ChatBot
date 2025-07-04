# 🧠 PDF Q&A Chatbot (Offline LLM + FAISS)

This project is a **Chat with your PDF** app that runs entirely **locally**, letting you upload lecture notes, manuals, or research papers and ask questions from them. It uses:
- 🧠 Local embedding models from Hugging Face (`sentence-transformers`)
- 🔍 Fast similarity search with FAISS
- 💬 Question answering powered by a local/remote LLM (OpenAI or HF)
- 🖥️ A user-friendly Streamlit interface

> ✅ No need to upload sensitive data to the cloud — everything runs on your machine (with optional OpenAI fallback).

---

## 📦 Features

- ✅ Upload any PDF (notes, textbooks, etc.)
- ✅ Automatic text chunking and vector embedding
- ✅ Question answering using Retrieval-Augmented Generation (RAG)
- ✅ Switch between local Hugging Face models or OpenAI GPT
- ✅ Lightweight UI built with Streamlit
- ✅ GPU support for local models (optional)

---

## 🖼️ Demo

![Demo Screenshot](demo/demo.png) <!-- Add your own screenshot here -->

---

## 🧱 Architecture

```bash
PDF ➜ PyPDF2 ➜ Text ➜ Chunked ➜ FAISS Index ➜ User Query
        ↘                        ↙
       SentenceTransformer Embeddings
                     ↘
       Retriever + LLM (HF or OpenAI)
