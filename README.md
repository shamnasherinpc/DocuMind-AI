# 📄 DocuMind AI - RAG Based PDF Chatbot

## 🚀 Project Overview
DocuMind AI is a Retrieval-Augmented Generation (RAG) based chatbot that allows users to ask questions from PDF documents. The system reads the PDF, processes the content, and provides accurate answers using a local AI model.

---

## 🧠 Features
- 📄 Reads and processes PDF documents
- 🤖 AI-powered question answering
- 🔍 Retrieves relevant context from documents
- 📌 Displays source page numbers
- 💻 Runs locally (no API key required)

---

## 🛠️ Technologies Used
- Python
- LangChain
- ChromaDB
- Ollama (Mistral Model)
- Sentence Transformers

---

## ⚙️ How It Works
1. Load PDF document
2. Split text into chunks
3. Convert text into embeddings
4. Store in vector database (ChromaDB)
5. Retrieve relevant data based on query
6. Generate answer using local LLM

---

## ▶️ How to Run

### Step 1: Install dependencies
```bash
pip install langchain langchain-community langchain-text-splitters chromadb sentence-transformers pypdf


## Citations Feature Added
- Now supports page-level citations in answers