import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DOCS_DIR = "docs"
CHROMA_DB_DIR = "chroma_db"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 200

def load_documents():
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.md",
        loader_cls=TextLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from {DOCS_DIR}")
    return documents

def split_documents(documents):
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks):
    """Create and persist Chroma vector store."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    print(f"Created vector store with {len(chunks)} chunks")
    print(f"Persisted to {CHROMA_DB_DIR}")
    return vector_store

def main():
    """Main function to load data into Chroma."""
    print("Starting data loading process...")

    documents = load_documents()

    if not documents:
        print("No documents found. Exiting.")
        return

    # we are splitting documents so we do not have a large chunk
    chunks = split_documents(documents)

    create_vector_store(chunks)

    print("Data loading complete!")

if __name__ == "__main__":
    main()
