from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
CHROMA_DB_DIR = "chroma_db"

def get_retriever(k=3):
    # Initialize embeddings (must use same model as load_data.py)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Load existing vector store
    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )

    # Create retriever
    retriever = vector_store.as_retriever(
        search_kwargs={"k": k}
    )

    return retriever

def search_documents(query, k=3):
    retriever = get_retriever(k=k)
    documents = retriever.invoke(query)
    return documents
