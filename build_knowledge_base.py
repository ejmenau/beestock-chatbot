# build_knowledge_base.py
import os
import re
import json
import numpy as np
import faiss
from docx import Document
from sentence_transformers import SentenceTransformer

def load_word_docs(directory_path):
    """Loads all .docx files from a directory and extracts their text and customer metadata."""
    docs_data = []
    print(f"\nScanning for Word documents in: {directory_path}")
    for filename in os.listdir(directory_path):
        if filename.endswith(".docx"):
            filepath = os.path.join(directory_path, filename)
            match = re.search(r"Processo(?:s)?\s(.*?)\s-", filename, re.IGNORECASE)
            customer_name = match.group(1) if match else "Unknown"
            
            doc = Document(filepath)
            full_text = "\n".join([para.text for para in doc.paragraphs if para.text])
            
            docs_data.append({
                "text": full_text,
                "metadata": {
                    "customer": customer_name,
                    "source_file": filename
                }
            })
    print(f"-> Found and loaded {len(docs_data)} Word documents.")
    return docs_data

def load_markdown_docs(directory_path):
    """Loads all .md files from a directory and extracts their text and topic metadata."""
    docs_data = []
    print(f"\nScanning for Markdown documents in: {directory_path}")
    for filename in os.listdir(directory_path):
        if filename.endswith(".md"):
            filepath = os.path.join(directory_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                full_text = f.read()
            
            topic_name = os.path.splitext(filename)[0]
            
            docs_data.append({
                "text": full_text,
                "metadata": {
                    "topic": topic_name,
                    "source_file": filename
                }
            })
    print(f"-> Found and loaded {len(docs_data)} Markdown documents.")
    return docs_data

def chunk_documents(documents_data):
    """Splits a list of documents into smaller chunks based on paragraphs."""
    all_chunks = []
    for doc_data in documents_data:
        text_chunks = doc_data["text"].split('\n\n')
        for i, chunk_text in enumerate(text_chunks):
            if chunk_text.strip():
                chunk_metadata = doc_data["metadata"].copy()
                chunk_metadata["chunk_id"] = f"{doc_data['metadata']['source_file']}_{i}"
                all_chunks.append({
                    "text_chunk": chunk_text.strip(),
                    "metadata": chunk_metadata
                })
    return all_chunks

def build_and_save_knowledge_base(chunks, model, index_path="vector_index.faiss", data_path="chunks_data.json"):
    """Generates embeddings and saves the complete knowledge base."""
    print(f"\nSplitting {len(chunks)} total chunks for processing.")
    print("Generating embeddings for all content (this may take a moment)...")
    
    texts_to_embed = [chunk["text_chunk"] for chunk in chunks]
    embeddings = model.encode(texts_to_embed, show_progress_bar=True)
    
    print("Normalizing embeddings and building FAISS index...")
    faiss.normalize_L2(embeddings)
    dimensionality = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimensionality)
    index.add(embeddings)
    
    print("Saving knowledge base files...")
    faiss.write_index(index, index_path)
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)

    print("\n============================================================")
    print("✅ Knowledge Base built successfully!")
    print("============================================================")
    print(f"Total vectors in index: {index.ntotal}")
    print(f"Total chunks in data file: {len(chunks)}")

if __name__ == "__main__":
    # --- CONFIGURE YOUR FOLDERS HERE ---
    WORD_DOCS_DIR = r"C:\Users\ejmen\Dropbox\Cursor\KB BeeStock"
    MARKDOWN_DIR = r"C:\Users\ejmen\Dropbox\BeeStock USA\KB Prototype\Markdown Files\Beestock"
    MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'

    # --- EXECUTION PIPELINE ---
    print("--- Starting Unified Knowledge Base Builder ---")
    
    # 1. Load the model
    print("Loading sentence transformer model...")
    model = SentenceTransformer(MODEL_NAME)
    
    # 2. Load all documents from all sources
    word_docs = load_word_docs(WORD_DOCS_DIR)
    markdown_docs = load_markdown_docs(MARKDOWN_DIR)
    all_docs = word_docs + markdown_docs
    
    # 3. Chunk all documents together
    all_chunks = chunk_documents(all_docs)
    
    # 4. Build and save the single, unified knowledge base
    build_and_save_knowledge_base(all_chunks, model)

