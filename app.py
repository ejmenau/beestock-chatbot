# app.py
import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os

# Configure Gemini API
genai.configure(api_key="AIzaSyApjah1pIDAbpG7O2guSi1UcqFKwrEo7hs")

@st.cache_resource
def load_rag_system():
    """Load the RAG system with error handling for missing files."""
    
    # Check if required files exist
    required_files = ['vector_index.faiss', 'chunks_data.json']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        st.error(f"❌ Missing required files: {', '.join(missing_files)}")
        st.info("""
        **To fix this issue:**
        
        1. **Run the indexing script locally first:**
           ```bash
           python build_index.py
           ```
        
        2. **Make sure these files are generated:**
           - `chunks_data.json`
           - `vector_index.faiss`
        
        3. **Include them in your deployment or run the indexing on the server.**
        """)
        return None, None, None
    
    try:
        # Load the FAISS vector index
        index = faiss.read_index('vector_index.faiss')
        
        # Load the chunks data
        with open('chunks_data.json', 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        # Load the sentence transformer model
        embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        
        return index, chunks_data, embedding_model
        
    except Exception as e:
        st.error(f"❌ Error loading RAG system: {str(e)}")
        return None, None, None

def search_knowledge_base(query, index, chunks_data, embedding_model, k=5, source_type_filter=None):
    """Search the knowledge base for relevant chunks."""
    try:
        # Generate embedding for the query
        query_embedding = embedding_model.encode([query])
        
        # Normalize the query embedding for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search the FAISS index
        search_k = min(k * 3, len(chunks_data))
        distances, indices = index.search(query_embedding, search_k)
        
        # Retrieve the corresponding chunks
        relevant_chunks = []
        for idx in indices[0]:
            if idx < len(chunks_data):
                chunk = chunks_data[idx]
                
                # Apply source type filter if specified
                if source_type_filter:
                    if chunk.get('source_type') != source_type_filter:
                        continue
                
                relevant_chunks.append(chunk)
                
                if len(relevant_chunks) >= k:
                    break
        
        return relevant_chunks
        
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

def generate_answer(query, relevant_chunks):
    """Generate an answer using Gemini based on retrieved context."""
    try:
        if not relevant_chunks:
            return "I could not find any relevant information in the knowledge base to answer your question."
        
        # Combine retrieved chunks into context
        context_str = "\n\n---\n".join([chunk['text'] for chunk in relevant_chunks])
        
        # Create prompt template
        prompt_template = f"""You are an expert support analyst for BeeStock WMS (Warehouse Management System). 
Your role is to help users understand processes, procedures, and information related to BeeStock.

IMPORTANT: Answer the user's question based ONLY on the information provided in the context below. 
Do not use any external knowledge or make assumptions beyond what is stated in the context.
If the context doesn't contain enough information to fully answer the question, say so clearly.

Context information:
{context_str}

User Question: {query}

Please provide a clear, helpful answer based on the context above:"""

        # Generate response using Gemini Pro
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt_template)
        
        return response.text.strip()
        
    except Exception as e:
        return f"Sorry, I encountered an error while generating the response: {str(e)}."

def format_sources(chunks):
    """Format source information for display."""
    if not chunks:
        return ""
    
    source_groups = {}
    
    for chunk in chunks:
        source_type = chunk.get('source_type', 'unknown')
        if source_type not in source_groups:
            source_groups[source_type] = []
        
        source_info = {
            'filename': chunk.get('source_filename', 'Unknown'),
            'customer': chunk.get('customer_name', 'Unknown')
        }
        
        if chunk.get('title'):
            source_info['title'] = chunk['title']
        
        source_groups[source_type].append(source_info)
    
    # Format the sources
    sources_text = "\n\n---\n**Sources Used:**\n"
    
    for source_type, sources in source_groups.items():
        if source_type == 'wiki_documentation':
            sources_text += "\n📚 **Wiki Documentation:**\n"
            for source in sources:
                title = source.get('title', source['filename'])
                sources_text += f"  - {title}\n"
        else:
            sources_text += f"\n💬 **Customer Transcripts ({sources[0]['customer']}):**\n"
            for source in sources:
                sources_text += f"  - {source['filename']}\n"
    
    return sources_text

# Streamlit UI
st.set_page_config(
    page_title="BeeStock WMS RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 BeeStock WMS RAG Chatbot")
st.markdown("Ask questions about BeeStock WMS processes based on our knowledge base.")

# Load the RAG system
with st.spinner("Loading RAG system..."):
    index, chunks_data, embedding_model = load_rag_system()

if index is None or chunks_data is None or embedding_model is None:
    st.stop()

# Show knowledge base info
with st.expander("📊 Knowledge Base Information"):
    source_types = {}
    customer_names = set()
    
    for chunk in chunks_data:
        source_type = chunk.get('source_type', 'unknown')
        source_types[source_type] = source_types.get(source_type, 0) + 1
        
        if chunk.get('customer_name') != 'Wiki Documentation':
            customer_names.add(chunk.get('customer_name', 'Unknown'))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Source Types:**")
        for source_type, count in source_types.items():
            st.write(f"- {source_type}: {count} chunks")
    
    with col2:
        if customer_names:
            st.write("**Customers:**")
            for customer in sorted(customer_names):
                st.write(f"- {customer}")

# Search interface
st.markdown("---")

# Query input
query = st.text_input("💬 Ask your question:", placeholder="e.g., How do I consult EAN?")

# Search options
col1, col2 = st.columns([2, 1])

with col1:
    source_filter = st.selectbox(
        "🔍 Filter by source type:",
        ["All sources", "Wiki Documentation", "Customer Transcripts"],
        help="Choose which type of documents to search"
    )

with col2:
    k_results = st.slider("Number of results:", min_value=1, max_value=10, value=5)

# Convert filter to internal format
source_type_filter = None
if source_filter == "Wiki Documentation":
    source_type_filter = "wiki_documentation"
elif source_filter == "Customer Transcripts":
    source_type_filter = "customer_transcript"

# Search button
if st.button("🔍 Search Knowledge Base", type="primary"):
    if query.strip():
        with st.spinner("Searching knowledge base..."):
            # Search for relevant chunks
            relevant_chunks = search_knowledge_base(
                query, index, chunks_data, embedding_model, 
                k=k_results, source_type_filter=source_type_filter
            )
            
            if relevant_chunks:
                # Generate answer
                with st.spinner("Generating answer..."):
                    answer = generate_answer(query, relevant_chunks)
                
                # Display results
                st.markdown("---")
                st.markdown("### 🤖 Answer")
                st.write(answer)
                
                # Show sources
                sources_info = format_sources(relevant_chunks)
                st.markdown(sources_info)
                
                # Show raw chunks for debugging
                with st.expander("🔍 View Retrieved Chunks"):
                    for i, chunk in enumerate(relevant_chunks):
                        st.markdown(f"**Chunk {i+1}** (from {chunk.get('source_filename', 'Unknown')})")
                        st.text(chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'])
                        st.markdown("---")
            else:
                st.warning("No relevant information found. Try rephrasing your question or checking different source types.")
    else:
        st.warning("Please enter a question to search.")

# Footer
st.markdown("---")
st.markdown("""
**💡 Tips:**
- Try different phrasings if you don't get the expected results
- Use the source type filter to focus on specific document types
- Check the retrieved chunks to see exactly what information was found
""")
