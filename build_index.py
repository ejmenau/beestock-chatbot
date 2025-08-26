# build_index.py
import os
import re
import json
import numpy as np
import faiss
from docx import Document
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def load_documents_from_directory(directory_path):
    """
    Load all .docx and .md files from the specified directory and extract text content.
    
    Args:
        directory_path (str): Path to the directory containing .docx and .md files
        
    Returns:
        list: List of dictionaries containing document text and metadata
    """
    documents = []
    
    # Check if directory exists
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    # Get all .docx and .md files in the directory
    docx_files = [f for f in os.listdir(directory_path) if f.endswith('.docx')]
    md_files = [f for f in os.listdir(directory_path) if f.endswith('.md')]
    
    all_files = docx_files + md_files
    
    if not all_files:
        print(f"No .docx or .md files found in {directory_path}")
        return documents
    
    print(f"Found {len(docx_files)} .docx files and {len(md_files)} .md files")
    
    # Process .docx files (customer transcripts)
    for filename in docx_files:
        file_path = os.path.join(directory_path, filename)
        
        try:
            # Extract customer name using regex
            # Pattern: "Processo [CustomerName] - Beestock" -> extract "CustomerName"
            customer_match = re.search(r'Processo[s]?\s+(.+?)\s+-\s+Beestock', filename)
            customer_name = customer_match.group(1) if customer_match else "Unknown"
            
            # Read the Word document
            doc = Document(file_path)
            
            # Extract text from all paragraphs
            full_text = ""
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():  # Only add non-empty paragraphs
                    full_text += paragraph.text.strip() + "\n\n"
            
            # Remove trailing newlines
            full_text = full_text.strip()
            
            if full_text:  # Only add documents with content
                documents.append({
                    'text': full_text,
                    'customer_name': customer_name,
                    'source_filename': filename,
                    'file_path': file_path,
                    'file_type': 'docx',
                    'source_type': 'customer_transcript'
                })
                
                print(f"  Loaded: {filename} (Customer: {customer_name})")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    # Process .md files (wiki documentation)
    for filename in md_files:
        file_path = os.path.join(directory_path, filename)
        
        try:
            # Read the Markdown file
            with open(file_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
            
            # Clean up the text (remove excessive whitespace but preserve structure)
            full_text = re.sub(r'\n{3,}', '\n\n', full_text)
            full_text = full_text.strip()
            
            if full_text:  # Only add documents with content
                # Extract a meaningful title from the filename or content
                title = filename.replace('.md', '').replace('_', ' ').replace('-', ' ')
                
                documents.append({
                    'text': full_text,
                    'customer_name': 'Wiki Documentation',
                    'source_filename': filename,
                    'file_path': file_path,
                    'file_type': 'md',
                    'source_type': 'wiki_documentation',
                    'title': title
                })
                
                print(f"  Loaded: {filename} (Type: Wiki Documentation)")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    print(f"Successfully loaded {len(documents)} documents")
    return documents

def chunk_documents(documents):
    """
    Split documents into smaller chunks based on paragraphs or sections.
    
    Args:
        documents (list): List of document dictionaries from load_documents_from_directory
        
    Returns:
        list: List of chunk dictionaries with text and metadata
    """
    chunks = []
    
    print("Chunking documents...")
    
    for doc in documents:
        if doc['file_type'] == 'docx':
            # For .docx files, split by double newlines (paragraphs)
            text_chunks = doc['text'].split('\n\n')
        else:
            # For .md files, split by headers and sections for better context
            # Split by markdown headers (lines starting with #)
            text_chunks = re.split(r'\n(?=#{1,6}\s)', doc['text'])
            
            # If no headers found, fall back to paragraph splitting
            if len(text_chunks) == 1:
                text_chunks = doc['text'].split('\n\n')
        
        for i, chunk_text in enumerate(text_chunks):
            chunk_text = chunk_text.strip()
            
            # Only add non-empty chunks with sufficient content
            if chunk_text and len(chunk_text) > 20:  # Increased minimum length for better context
                chunk = {
                    'text': chunk_text,
                    'customer_name': doc['customer_name'],
                    'source_filename': doc['source_filename'],
                    'file_path': doc['file_path'],
                    'file_type': doc['file_type'],
                    'source_type': doc['source_type'],
                    'chunk_index': i,
                    'total_chunks': len(text_chunks)
                }
                
                # Add title for markdown files
                if 'title' in doc:
                    chunk['title'] = doc['title']
                
                chunks.append(chunk)
    
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks

def generate_embeddings(chunks):
    """
    Generate embeddings for all text chunks using SentenceTransformer.
    
    Args:
        chunks (list): List of chunk dictionaries
        
    Returns:
        numpy.ndarray: Array of embeddings
    """
    print("Generating embeddings...")
    
    # Initialize the SentenceTransformer model
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    
    # Extract text from chunks
    texts = [chunk['text'] for chunk in chunks]
    
    # Generate embeddings with progress bar
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=False)
    
    print(f"Generated embeddings with shape: {embeddings.shape}")
    return embeddings

def build_and_save_index(chunks, embeddings, chunks_file='chunks_data.json', index_file='vector_index.faiss'):
    """
    Build FAISS index and save both chunks data and vector index.
    
    Args:
        chunks (list): List of chunk dictionaries
        embeddings (numpy.ndarray): Array of embeddings
        chunks_file (str): Output JSON file for chunks data
        index_file (str): Output FAISS index file
    """
    print("Building and saving index...")
    
    # Save chunks data to JSON
    print(f"Saving chunks data to {chunks_file}...")
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    # Create FAISS index
    print("Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    # Normalize embeddings for cosine similarity (L2 normalization)
    print("Normalizing embeddings for cosine similarity...")
    faiss.normalize_L2(embeddings)
    
    # Add embeddings to the index
    index.add(embeddings.astype('float32'))
    
    # Save FAISS index
    print(f"Saving FAISS index to {index_file}...")
    faiss.write_index(index, index_file)
    
    print(f"Index built successfully!")
    print(f"  - Total vectors: {index.ntotal}")
    print(f"  - Vector dimension: {index.d}")
    print(f"  - Index type: {type(index).__name__}")

def check_and_install_dependencies():
    """Check if required packages are installed and install if missing."""
    required_packages = {
        'python-docx': 'docx',
        'sentence-transformers': 'sentence_transformers',
        'faiss-cpu': 'faiss',
        'numpy': 'numpy',
        'tqdm': 'tqdm'
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name} is available")
        except ImportError:
            missing_packages.append(package_name)
            print(f"✗ {package_name} is missing")
    
    if missing_packages:
        print(f"\nInstalling missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            os.system(f"pip install {package}")
        print("Dependencies installed. Please restart the script if needed.")
        return False
    
    return True

if __name__ == "__main__":
    # Check dependencies first
    if not check_and_install_dependencies():
        print("Please restart the script after installing dependencies.")
        exit(1)
    
    # Define the knowledge base directory path
    kb_directory = r"C:\Users\ejmen\Dropbox\Cursor\KB BeeStock"
    
    print("=" * 60)
    print("RAG Chatbot - Indexing Pipeline")
    print("=" * 60)
    print(f"Knowledge Base Directory: {kb_directory}")
    print()
    
    try:
        # Step 1: Load documents
        print("Step 1: Loading documents...")
        documents = load_documents_from_directory(kb_directory)
        
        if not documents:
            print("No documents found. Exiting.")
            exit(1)
        
        print()
        
        # Step 2: Chunk documents
        print("Step 2: Chunking documents...")
        chunks = chunk_documents(documents)
        
        if not chunks:
            print("No chunks created. Exiting.")
            exit(1)
        
        print()
        
        # Step 3: Generate embeddings
        print("Step 3: Generating embeddings...")
        embeddings = generate_embeddings(chunks)
        
        print()
        
        # Step 4: Build and save index
        print("Step 4: Building and saving index...")
        build_and_save_index(chunks, embeddings)
        
        print()
        print("=" * 60)
        print("Indexing Pipeline completed successfully!")
        print("=" * 60)
        print(f"Output files created:")
        print(f"  - chunks_data.json: Contains all text chunks with metadata")
        print(f"  - vector_index.faiss: FAISS vector index for similarity search")
        print()
        print("You can now use these files with your RAG chatbot!")
        
    except Exception as e:
        print(f"Error during indexing pipeline: {str(e)}")
        print("Please check the error message and try again.")
        exit(1)

