from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

def build_vector_store(texts: list[str]):
    """
    Converts raw texts into embeddings and stores them in FAISS.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents(texts)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)

    return vector_store

def retrieve_relevant_chunks(vector_store, query: str, k: int = 4):
    """
    Retrieves top-k semantically relevant chunks for a query.
    """
    return vector_store.similarity_search(query, k=k)

