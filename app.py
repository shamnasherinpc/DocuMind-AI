import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
import tempfile

st.set_page_config(page_title="DocuMind AI", layout="wide")

st.title("📄 DocuMind AI - Chat with your PDF")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("✅ PDF uploaded successfully!")

    # Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Vector DB
    vectorstore = Chroma.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    # Load model
    llm = Ollama(model="mistral")

    st.subheader("💬 Ask Questions")

    query = st.text_input("Enter your question:")

    if query:
        with st.spinner("Thinking... 🤖"):

            relevant_docs = retriever.invoke(query)

            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            prompt = f"""
You are an AI assistant.

Answer ONLY using the context below.
If the answer is not in the context, say: "I don't know."

Context:
{context}

Question:
{query}

Answer:
"""

            response = llm.invoke(prompt)

            st.markdown("### 📌 Answer")
            st.write(response)

            st.markdown("### 📄 Sources")
            for doc in relevant_docs:
                st.write(f"Page: {doc.metadata.get('page', 'N/A')}")