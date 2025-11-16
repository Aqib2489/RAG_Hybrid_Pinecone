import os
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

# LangChain core
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Models & embeddings
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings

# Load PDFs
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# In-memory BM25 retriever
from langchain_community.retrievers import BM25Retriever


# ==============================
# 1. CONFIG
# ==============================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")  # e.g. your new 1024-dim index
PDF_FOLDER = os.getenv("PDF_FOLDER", "./pdfs")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set")
if not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME not set")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


# ==============================
# 2. LOAD & SPLIT PDFs
# ==============================

def load_and_split_pdfs(pdf_folder: str) -> List[Document]:
    folder = Path(pdf_folder)
    folder.mkdir(parents=True, exist_ok=True)

    docs: List[Document] = []
    for pdf in folder.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()
        for d in pages:
            d.metadata["source"] = pdf.name
            d.metadata["file_path"] = str(pdf)
        docs.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    print(f"[PDF] Loaded & split into {len(chunks)} chunks.")
    return chunks


# ==============================
# 3. HYBRID RETRIEVER
# ==============================

class HybridRetriever(BaseRetriever):
    """
    Combines:
      - dense semantic retrieval (Pinecone)
      - BM25 keyword retrieval (in-memory)
    """

    vector_retriever: Any
    bm25_retriever: Any
    k: int = 6

    def _get_relevant_documents(self, query: str) -> List[Document]:
        dense_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        merged: List[Document] = []
        seen = set()

        def add(docs, tag):
            for d in docs:
                key = (d.page_content, d.metadata.get("source"))
                if key in seen:
                    continue
                seen.add(key)
                d.metadata["retriever"] = tag
                merged.append(d)
                if len(merged) >= self.k:
                    return

        add(dense_docs, "dense")
        add(bm25_docs, "bm25")
        return merged


# ==============================
# 4. BUILD RAG CHAIN
# ==============================

def build_rag_chain(docs: List[Document]):
    # Embeddings must match your Pinecone index (llama-text-embed-v2 → 1024-dim index)
    embeddings = PineconeEmbeddings(model="llama-text-embed-v2")

    # Pinecone vector store (existing index)
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
    )

    # Upsert docs to Pinecone
    if docs:
        vector_store.add_documents(docs)
        print(f"[Pinecone] Upserted {len(docs)} chunks.")
    else:
        print("[WARN] No documents to index into Pinecone.")

    # In-memory BM25 over the same docs
    bm25_retriever = BM25Retriever.from_documents(docs)

    # Hybrid retriever
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        k=6,
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    # Step 1: retrieve + prepare context
    def retrieve_and_prepare(inputs: Dict[str, Any]) -> Dict[str, Any]:
        query = inputs["input"]
        chat_history = inputs.get("chat_history", [])
        docs = hybrid_retriever.invoke(query)
        context = "\n\n".join(d.page_content for d in docs)
        return {
            "context": context,
            "input": query,
            "chat_history": chat_history,
        }

    prepare = RunnableLambda(retrieve_and_prepare)

    # Step 2: prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant for question-answering over PDF documents.\n"
                "Use the provided context to answer the question.\n"
                "If the answer is not in the context, say you don't know.\n"
                "Be concise.",
            ),
            ("system", "Context:\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    chain = prepare | prompt | llm | StrOutputParser()

    # Conversation memory
    store: Dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    conversational_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return conversational_chain


# ==============================
# 5. MAIN CLI
# ==============================

def main():
    docs = load_and_split_pdfs(PDF_FOLDER)
    rag = build_rag_chain(docs)

    session_id = "user-1"
    print("Hybrid RAG (Pinecone + in-memory BM25) with conversation memory.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        answer = rag.invoke(
            {"input": q},
            config={"configurable": {"session_id": session_id}},
        )
        print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    main()
