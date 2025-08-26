import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

class RAGSystem:
    """
    RAG (Retrieval-Augmented Generation) system for BeeStock WMS support.
    Loads pre-built vector index and provides question-answering capabilities.
    """
    
    def __init__(self, index_file='vector_index.faiss', chunks_file='chunks_data.json'):
        """
        Initialize the RAG system by loading the vector index, chunks data, and sentence transformer model.
        
        Args:
            index_file (str): Path to the FAISS vector index file
            chunks_file (str): Path to the JSON file containing text chunks and metadata
        """
        print("Initializing RAG System...")
        
        # Load the FAISS vector index
        print("Loading vector index...")
        self.index = faiss.read_index(index_file)
        print(f"✓ Vector index loaded: {self.index.ntotal} vectors, {self.index.d} dimensions")
        
        # Load the chunks data
        print("Loading chunks data...")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            self.chunks_data = json.load(f)
        print(f"✓ Chunks data loaded: {len(self.chunks_data)} chunks")
        
        # Load the sentence transformer model
        print("Loading sentence transformer model...")
        self.model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        print("✓ Sentence transformer model loaded")
        
        # Analyze the knowledge base
        self._analyze_knowledge_base()
        
        print("RAG System initialization complete! 🚀")
        print()
    
    def _analyze_knowledge_base(self):
        """Analyze the loaded knowledge base to show what's available."""
        source_types = {}
        customer_names = set()
        
        for chunk in self.chunks_data:
            source_type = chunk.get('source_type', 'unknown')
            source_types[source_type] = source_types.get(source_type, 0) + 1
            
            if chunk.get('customer_name') != 'Wiki Documentation':
                customer_names.add(chunk.get('customer_name', 'Unknown'))
        
        print("📊 Knowledge Base Analysis:")
        for source_type, count in source_types.items():
            print(f"  - {source_type}: {count} chunks")
        
        if customer_names:
            print(f"  - Customers: {', '.join(sorted(customer_names))}")
        print()
    
    def search(self, query, k=5, customer_filter=None, source_type_filter=None):
        """
        Search for the most relevant text chunks based on the user's query.
        
        Args:
            query (str): The user's question
            k (int): Number of top results to return (default: 5)
            customer_filter (str, optional): Filter results by specific customer name
            source_type_filter (str, optional): Filter by source type ('customer_transcript' or 'wiki_documentation')
            
        Returns:
            list: List of relevant text chunks with metadata
        """
        # Generate embedding for the query
        query_embedding = self.model.encode([query])
        
        # Normalize the query embedding for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search the FAISS index with more results to allow for filtering
        search_k = min(k * 3, len(self.chunks_data))  # Get more results to filter from
        distances, indices = self.index.search(query_embedding, search_k)
        
        # Retrieve the corresponding chunks
        relevant_chunks = []
        for idx in indices[0]:
            if idx < len(self.chunks_data):  # Ensure index is valid
                chunk = self.chunks_data[idx]
                
                # Apply customer filter if specified
                if customer_filter and chunk.get('customer_name') != 'Wiki Documentation':
                    if chunk['customer_name'].lower() != customer_filter.lower():
                        continue
                
                # Apply source type filter if specified
                if source_type_filter:
                    if chunk.get('source_type') != source_type_filter:
                        continue
                
                relevant_chunks.append(chunk)
                
                # Stop when we have enough results
                if len(relevant_chunks) >= k:
                    break
        
        return relevant_chunks
    
    def ask(self, query, customer_filter=None, source_type_filter=None):
        """
        Main RAG method: retrieve relevant context and generate an answer using Gemini.
        
        Args:
            query (str): The user's question
            customer_filter (str, optional): Filter results by specific customer name
            source_type_filter (str, optional): Filter by source type
            
        Returns:
            str: Generated answer from Gemini based on retrieved context
        """
        # Search for relevant context chunks
        relevant_chunks = self.search(query, k=5, customer_filter=customer_filter, source_type_filter=source_type_filter)
        
        if not relevant_chunks:
            return "I could not find any relevant information in the knowledge base to answer your question. Please try rephrasing your question or ask about a different topic."
        
        # Combine retrieved chunks into context
        context_str = "\n\n---\n".join([chunk['text'] for chunk in relevant_chunks])
        
        # Create detailed prompt template for Gemini
        prompt_template = f"""You are an expert support analyst for BeeStock WMS (Warehouse Management System). 
Your role is to help users understand processes, procedures, and information related to BeeStock.

IMPORTANT: Answer the user's question based ONLY on the information provided in the context below. 
Do not use any external knowledge or make assumptions beyond what is stated in the context.
If the context doesn't contain enough information to fully answer the question, say so clearly.

Context information:
{context_str}

User Question: {query}

Please provide a clear, helpful answer based on the context above:"""

        try:
            # Generate response using Gemini Pro
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt_template)
            
            # Extract the text response
            answer = response.text.strip()
            
            # Add detailed source information
            sources_info = self._format_sources(relevant_chunks)
            
            return answer + sources_info
            
        except Exception as e:
            return f"Sorry, I encountered an error while generating the response: {str(e)}. Please try again."
    
    def _format_sources(self, chunks):
        """Format source information for the response."""
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

def main():
    """Main execution function for the RAG chatbot."""
    
    # IMPORTANT: Replace 'YOUR_GEMINI_API_KEY' with your actual Gemini API key
    # You can get one from: https://makersuite.google.com/app/apikey
    genai.configure(api_key="AIzaSyApjah1pIDAbpG7O2guSi1UcqFKwrEo7hs")
    
    print("=" * 70)
    print("🤖 BeeStock WMS RAG Chatbot")
    print("=" * 70)
    print("This chatbot can answer questions about BeeStock WMS processes")
    print("based on the knowledge base of customer documentation and wiki articles.")
    print()
    
    try:
        # Initialize the RAG system
        rag_system = RAGSystem()
        
        print("Chat session started! Type 'exit' to quit.")
        print("-" * 50)
        
        # Main chat loop
        while True:
            try:
                # Get user input
                user_query = input("\n💬 Ask your question (or type 'exit' to quit): ").strip()
                
                # Check for exit command
                if user_query.lower() in ['exit', 'quit', 'bye']:
                    print("\n👋 Thank you for using the BeeStock WMS RAG Chatbot!")
                    break
                
                # Skip empty queries
                if not user_query:
                    print("Please enter a question.")
                    continue
                
                # Ask for search preferences
                print("\n🔍 Search Options:")
                print("1. Search everything (default)")
                print("2. Search only wiki documentation")
                print("3. Search only customer transcripts")
                print("4. Search specific customer")
                
                choice = input("Choose option (1-4, default: 1): ").strip()
                
                customer_filter = None
                source_type_filter = None
                
                if choice == "2":
                    source_type_filter = "wiki_documentation"
                    print("🔍 Searching only wiki documentation...")
                elif choice == "3":
                    source_type_filter = "customer_transcript"
                    print("🔍 Searching only customer transcripts...")
                elif choice == "4":
                    customer_filter = input("Enter customer name: ").strip()
                    if customer_filter:
                        print(f"🔍 Filtering results for customer: {customer_filter}")
                else:
                    print("🔍 Searching all sources...")
                
                print("\n🔍 Searching knowledge base...")
                
                # Get answer from RAG system
                answer = rag_system.ask(user_query, customer_filter, source_type_filter)
                
                # Display the answer
                print("\n" + "=" * 50)
                print("🤖 ANSWER:")
                print("=" * 50)
                print(answer)
                print("=" * 50)
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("Please try again or type 'exit' to quit.")
    
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find required files: {str(e)}")
        print("Please make sure you have run 'build_index.py' first to create the index files.")
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        print("Please check your setup and try again.")

if __name__ == "__main__":
    main()
