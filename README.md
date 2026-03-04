# 🏥 Medmate: Multimodal Medical RAG API

A robust, AI-powered FastAPI backend that processes medical PDFs and provides intelligent, multimodal answers (text + images) using a Retrieval-Augmented Generation (RAG) pipeline. 

This API uses **ChromaDB** for local vector storage, **LangChain** for document processing, and **OpenRouter** to intelligently route queries to state-of-the-art multimodal Large Language Models (like Gemini 2.0 Flash and Gemma 3), complete with an automated fallback mechanism to ensure maximum uptime.

## ✨ Features
* **📄 PDF Knowledge Base:** Upload medical PDFs, automatically chunk them, and store them locally in a Chroma vector database using HuggingFace embeddings.
* **👁️ Multimodal Chat:** Ask medical questions using text and optional image uploads (e.g., uploading a photo of a rash alongside a text query).
* **🔄 Automated LLM Fallback:** Gracefully falls back through a curated list of OpenRouter models if the primary model goes offline or hits rate limits.
* **🧱 Structured JSON Responses:** The AI is strictly prompted to return clean, parsed JSON objects containing `disease` and `firstaid` categories for easy frontend integration.

## 🛠️ Tech Stack
* **Framework:** FastAPI, Uvicorn
* **RAG Pipeline:** LangChain, HuggingFace (`all-MiniLM-L6-v2`), ChromaDB
* **LLM Integration:** OpenAI Python SDK (configured for OpenRouter API)

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Python 3.9+** installed on your machine.

### 2. Installation
Clone the repository and set up a virtual environment:

```bash
# Clone the repo
git clone [https://github.com/yourusername/medmate-rag-api.git](https://github.com/yourusername/medmate-rag-api.git)
cd medmate-rag-api

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install the required dependencies
pip install -r requirements.txt
