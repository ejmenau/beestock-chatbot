# add_markdown.py
import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def load_existing_data(index_path="vector_index.faiss", data_path="chunks_data.json"):
    """Loads the existing FAISS index and chunk data if they exist."""
    if os.path.exists(index_path) and os.path.exists(data_path):
        print("Found existing knowledge base. Loading...")
        index = faiss.read_index(index_path)
        with open(data_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        print(f"✓ Loaded {index.ntotal} vectors and {len(chunks_data)} chunks.")
        return index, chunks_data
    else:
        print("No existing knowledge base found. A new one will be created.")
        return None, []

def load_markdown_from_directory(directory_path):
    """Loads all .md files from a directory and extracts their text."""
    markdown_data = []
    print(f"\nSearching for .md files in: {directory_path}")
    for filename in os.listdir(directory_path):
        if filename.endswith(".md"):
            filepath = os.path.join(directory_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                full_text = f.read()
            
            # Use the filename (without extension) as the primary metadata 'topic'
            topic_name = os.path.splitext(filename)[0]
            
            markdown_data.append({
                "text": full_text,
                "metadata": {
                    "topic": topic_name,
                    "source_file": filename
                }
            })
    print(f"Found and loaded {len(markdown_data)} markdown documents.")
    return markdown_data

def chunk_documents(documents_data):
    """Splits documents into smaller chunks based on paragraphs."""
    all_chunks = []
    for doc_data in documents_data:
        # Split by double newline, which is a common paragraph separator in Markdown
        text_chunks = doc_data["text"].split('\n\n')
        
        for i, chunk_text in enumerate(text_chunks):
            if chunk_text.strip():
                chunk_metadata = doc_data["metadata"].copy()
                chunk_metadata["chunk_id"] = f"{doc_data['metadata']['source_file']}_{i}"
                
                all_chunks.append({
                    "text_chunk": chunk_text.strip(),
                    "metadata": chunk_metadata
                })
    print(f"Split new documents into {len(all_chunks)} chunks.")
    return all_chunks

def generate_embeddings(chunks, model):
    """Generates vector embeddings for a list of text chunks."""
    texts_to_embed = [chunk["text_chunk"] for chunk in chunks]
    embeddings = model.encode(texts_to_embed, show_progress_bar=True)
    return embeddings

def save_updated_knowledge_base(index, chunks, index_path="vector_index.faiss", data_path="chunks_data.json"):
    """Saves the updated index and chunk data to disk."""
    print("\nSaving updated knowledge base...")
    faiss.write_index(index, index_path)
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
    
    print("\n============================================================")
    print("Knowledge Base updated successfully!")
    print("============================================================")
    print(f"Total vectors in index: {index.ntotal}")
    print(f"Total chunks in data file: {len(chunks)}")

if __name__ == "__main__":
    # --- SCRIPT CONFIGURATION ---
    MARKDOWN_DIR = r"C:\Users\ejmen\Dropbox\BeeStock USA\KB Prototype\Markdown Files\Beestock"
    MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'

    # --- EXECUTION PIPELINE ---
    print("Starting Knowledge Base Update Process...")
    
    # 1. Load the multilingual model
    print("\nLoading sentence transformer model...")
    model = SentenceTransformer(MODEL_NAME)
    print("✓ Model loaded.")

    # 2. Load existing index and data
    faiss_index, all_chunks_data = load_existing_data()

    # 3. Load new markdown files
    new_docs = load_markdown_from_directory(MARKDOWN_DIR)

    if new_docs:
        # 4. Chunk the new documents
        new_chunks = chunk_documents(new_docs)

        # 5. Generate embeddings for the NEW chunks only
        print("\nGenerating embeddings for new documents...")
        new_embeddings = generate_embeddings(new_chunks, model)
        faiss.normalize_L2(new_embeddings)

        # 6. Update the knowledge base
        if faiss_index is None:
            # This is the first run, create a new index
            dimensionality = new_embeddings.shape[1]
            faiss_index = faiss.IndexFlatL2(dimensionality)
        
        faiss_index.add(new_embeddings)
        all_chunks_data.extend(new_chunks)

        # 7. Save the combined data
        save_updated_knowledge_base(faiss_index, all_chunks_data)
    else:
        print("\nNo new markdown files found to add.")

