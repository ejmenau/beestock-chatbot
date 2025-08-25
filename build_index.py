# build_index.py
import os
import re
import json
import numpy as np
import faiss
from docx import Document
from sentence_transformers import SentenceTransformer
import sys
import subprocess

def install_missing_packages():
    """Checks for and installs any missing required packages."""
    required_packages = {
        "python-docx": "python-docx",
        "sentence-transformers": "sentence-transformers",
        "faiss-cpu": "faiss-cpu",
        "numpy": "numpy"
    }
    for package_name, install_name in required_packages.items():
        try:
            __import__(package_name)
        except ImportError:
            print(f"Installing {install_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])

def load_documents_from_directory(directory_path):
    """
    Loads all .docx files from a directory, extracts text, and associates
    metadata (customer name) from the filename.
    """
    documents_data = []
    print(f"Searching for .docx files in: {directory_path}")
    for filename in os.listdir(directory_path):
        if filename.endswith(".docx"):
            filepath = os.path.join(directory_path, filename)
            # Extract customer name from filename (e.g., "Processo BeautyColor -...")
            match = re.search(r"Processo(?:s)?\s(.*?)\s-", filename, re.IGNORECASE)
            customer_name = match.group(1) if match else "Unknown"
            
            doc = Document(filepath)
            full_text = "\n".join([para.text for para in doc.paragraphs if para.text])
            
            documents_data.append({
                "text": full_text,
                "metadata": {
                    "customer": customer_name,
                    "source_file": filename
                }
            })
    print(f"Found and loaded {len(documents_data)} documents.")
    return documents_data

def chunk_documents(documents_data):
    """
    Splits documents into smaller chunks based on paragraphs (\\n\\n)
    and attaches metadata to each chunk.
    """
    all_chunks = []
    for doc_data in documents_data:
        # A simple recursive strategy: split by double newline (paragraphs)
        text_chunks = doc_data["text"].split('\n\n')
        
        for i, chunk_text in enumerate(text_chunks):
            if chunk_text.strip():  # Ensure chunk is not just whitespace
                chunk_metadata = doc_data["metadata"].copy()
                chunk_metadata["chunk_id"] = f"{doc_data['metadata']['source_file']}_{i}"
                
                all_chunks.append({
                    "text_chunk": chunk_text.strip(),
                    "metadata": chunk_metadata
                })
    print(f"Split documents into {len(all_chunks)} chunks.")
    return all_chunks

def generate_embeddings(chunks, model):
    """
    Generates vector embeddings for a list of text chunks using the provided model.
    """
    texts_to_embed = [chunk["text_chunk"] for chunk in chunks]
    embeddings = model.encode(texts_to_embed, show_progress_bar=True)
    return embeddings

def build_and_save_index(chunks, embeddings, index_path="vector_index.faiss", data_path="chunks_data.json"):
    """
    Saves chunks and metadata to a JSON file and builds/saves a FAISS index.
    """
    # Save the chunks and metadata
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
        
    # Build the FAISS index
    dimensionality = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimensionality)
    
    # FAISS requires normalized vectors for efficient cosine similarity search
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    # Save the index to disk
    faiss.write_index(index, index_path)
    
    print("\n============================================================")
    print("Indexing Pipeline completed successfully!")
    print("============================================================")
    print("Output files created:")
    print(f"  - {data_path}: Contains all text chunks with metadata")
    print(f"  - {index_path}: FAISS vector index for similarity search")
    print("\nYou can now use these files with your RAG chatbot!")


if __name__ == "__main__":
    install_missing_packages()

    # --- SCRIPT CONFIGURATION ---
    # THE ONLY LINE YOU NEED TO CHANGE: Point this to the folder with your .docx files
    KNOWLEDGE_BASE_DIR = r"C:\Users\ejmen\Dropbox\Cursor\KB BeeStock"
    
    # **FIX**: Switched to a powerful multilingual model
    MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'

    # --- EXECUTION PIPELINE ---
    print("Starting Indexing Pipeline...")
    print(f"Using model: {MODEL_NAME}")
    
    # 1. Initialize the embedding model
    print("\nLoading sentence transformer model...")
    model = SentenceTransformer(MODEL_NAME)
    print("✓ Model loaded.")

    # 2. Load documents
    print("\nLoading documents...")
    documents = load_documents_from_directory(KNOWLEDGE_BASE_DIR)
    
    # 3. Chunk documents
    print("\nChunking documents...")
    text_chunks = chunk_documents(documents)
    
    # 4. Generate embeddings
    print("\nGenerating embeddings (this may take a moment)...")
    embeddings_matrix = generate_embeddings(text_chunks, model)
    
    # 5. Build and save the index
    print("\nBuilding and saving index...")
    build_and_save_index(text_chunks, embeddings_matrix)

