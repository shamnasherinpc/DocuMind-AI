from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

# -----------------------------
# Step 1: Load PDF
# -----------------------------
loader = PyPDFLoader("sample.pdf")
documents = loader.load()

# -----------------------------
# Step 2: Split Text
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# -----------------------------
# Step 3: Create Embeddings
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# -----------------------------
# Step 4: Store in Vector DB
# -----------------------------
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)

# -----------------------------
# Step 5: Load Local LLM
# -----------------------------
llm = Ollama(model="mistral")

# -----------------------------
# Step 6: Create Retriever
# -----------------------------
retriever = vectorstore.as_retriever()

# -----------------------------
# Step 7: Chat Loop
# -----------------------------
print("\n📄 DocuMind AI is ready! Ask questions from your PDF.\n")

while True:
    query = input("👉 Ask a question (or type 'exit'): ")

    if query.lower() == "exit":
        print("👋 Exiting...")
        break

    # ✅ FIXED LINE (latest LangChain)
    relevant_docs = retriever.invoke(query)

    # Combine context
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Prompt
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

    # Get response from Ollama
    response = llm.invoke(prompt)

    # Output
    print("\n" + "="*50)
    print("📌 ANSWER:")
    print("="*50)
    print(response)

    print("\n" + "-"*50)
    print("📄 SOURCES (Page Numbers):")
    for doc in relevant_docs:
        print(f"👉 Page: {doc.metadata.get('page', 'N/A')}")
    print("="*50)