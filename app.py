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
    """Carrega o sistema RAG com tratamento de erros para arquivos ausentes."""
    
    # Verifica se os arquivos necessários existem
    required_files = ['vector_index.faiss', 'chunks_data.json']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        st.error(f"❌ Arquivos necessários não encontrados: {', '.join(missing_files)}")
        st.info("""
        **Para resolver este problema:**
        
        1. **Execute o script de indexação localmente primeiro:**
           ```bash
           python build_index.py
           ```
        
        2. **Certifique-se de que estes arquivos foram gerados:**
           - `chunks_data.json`
           - `vector_index.faiss`
        
        3. **Inclua-os no seu deployment ou execute a indexação no servidor.**
        """)
        return None, None, None
    
    try:
        # Carrega o índice vetorial FAISS
        index = faiss.read_index('vector_index.faiss')
        
        # Carrega os dados dos chunks
        with open('chunks_data.json', 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        # Carrega o modelo de embedding
        embedding_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        
        return index, chunks_data, embedding_model
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar o sistema RAG: {str(e)}")
        return None, None, None

def search_knowledge_base(query, index, chunks_data, embedding_model, k=5, source_type_filter=None):
    """Busca na base de conhecimento por chunks relevantes."""
    try:
        # Gera embedding para a consulta
        query_embedding = embedding_model.encode([query])
        
        # Normaliza o embedding da consulta para similaridade de cosseno
        faiss.normalize_L2(query_embedding)
        
        # Busca no índice FAISS
        search_k = min(k * 3, len(chunks_data))
        distances, indices = index.search(query_embedding, search_k)
        
        # Recupera os chunks correspondentes
        relevant_chunks = []
        for idx in indices[0]:
            if idx < len(chunks_data):
                chunk = chunks_data[idx]
                
                # Aplica filtro de tipo de fonte se especificado
                if source_type_filter:
                    if chunk.get('source_type') != source_type_filter:
                        continue
                
                relevant_chunks.append(chunk)
                
                if len(relevant_chunks) >= k:
                    break
        
        return relevant_chunks
        
    except Exception as e:
        st.error(f"Erro na busca: {str(e)}")
        return []

def generate_answer(query, relevant_chunks):
    """Gera uma resposta usando Gemini baseada no contexto recuperado."""
    try:
        if not relevant_chunks:
            return "Não consegui encontrar informações relevantes na base de conhecimento para responder sua pergunta."
        
        # Combina os chunks recuperados em contexto
        context_str = "\n\n---\n".join([chunk['text'] for chunk in relevant_chunks])
        
        # Cria template de prompt em português
        prompt_template = f"""Você é um analista de suporte especialista no sistema WMS BeeStock (Sistema de Gerenciamento de Armazém). 
Seu papel é ajudar os usuários a entender processos, procedimentos e informações relacionadas ao BeeStock.

IMPORTANTE: Responda à pergunta do usuário baseado APENAS nas informações fornecidas no contexto abaixo. 
Não use conhecimento externo ou faça suposições além do que está declarado no contexto.
Se o contexto não contiver informações suficientes para responder completamente à pergunta, diga isso claramente.

Informações do contexto:
{context_str}

Pergunta do usuário: {query}

Por favor, forneça uma resposta clara e útil baseada no contexto acima:"""

        # Gera resposta usando Gemini Pro
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt_template)
        
        return response.text.strip()
        
    except Exception as e:
        return f"Desculpe, encontrei um erro ao gerar a resposta: {str(e)}."

def format_sources(chunks):
    """Formata informações das fontes para exibição."""
    if not chunks:
        return ""
    
    source_groups = {}
    
    for chunk in chunks:
        source_type = chunk.get('source_type', 'unknown')
        if source_type not in source_groups:
            source_groups[source_type] = []
        
        source_info = {
            'filename': chunk.get('source_filename', 'Desconhecido'),
            'customer': chunk.get('customer_name', 'Desconhecido')
        }
        
        if chunk.get('title'):
            source_info['title'] = chunk['title']
        
        source_groups[source_type].append(source_info)
    
    # Formata as fontes
    sources_text = "\n\n---\n**Fontes Utilizadas:**\n"
    
    for source_type, sources in source_groups.items():
        if source_type == 'wiki_documentation':
            sources_text += "\n📚 **Documentação Wiki:**\n"
            for source in sources:
                title = source.get('title', source['filename'])
                sources_text += f"  - {title}\n"
        else:
            sources_text += f"\n💬 **Transcrições de Cliente ({sources[0]['customer']}):**\n"
            for source in sources:
                sources_text += f"  - {source['filename']}\n"
    
    return sources_text

# Interface Streamlit
st.set_page_config(
    page_title="Chatbot RAG BeeStock WMS",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Chatbot RAG BeeStock WMS")
st.markdown("Faça perguntas sobre processos do BeeStock WMS baseado em nossa base de conhecimento.")

# Carrega o sistema RAG
with st.spinner("Carregando sistema RAG..."):
    index, chunks_data, embedding_model = load_rag_system()

if index is None or chunks_data is None or embedding_model is None:
    st.stop()

# Mostra informações da base de conhecimento
with st.expander("📊 Informações da Base de Conhecimento"):
    source_types = {}
    customer_names = set()
    
    for chunk in chunks_data:
        source_type = chunk.get('source_type', 'unknown')
        source_types[source_type] = source_types.get(source_type, 0) + 1
        
        if chunk.get('customer_name') != 'Documentação Wiki':
            customer_names.add(chunk.get('customer_name', 'Desconhecido'))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Tipos de Fonte:**")
        for source_type, count in source_types.items():
            if source_type == 'wiki_documentation':
                st.write(f"- Documentação Wiki: {count} chunks")
            elif source_type == 'customer_transcript':
                st.write(f"- Transcrições de Cliente: {count} chunks")
            else:
                st.write(f"- {source_type}: {count} chunks")
    
    with col2:
        if customer_names:
            st.write("**Clientes:**")
            for customer in sorted(customer_names):
                st.write(f"- {customer}")

# Interface de busca
st.markdown("---")

# Entrada da consulta
query = st.text_input("💬 Faça sua pergunta:", placeholder="ex: Como consultar EAN?")

# Opções de busca
col1, col2 = st.columns([2, 1])

with col1:
    source_filter = st.selectbox(
        "🔍 Filtrar por tipo de fonte:",
        ["Todas as fontes", "Documentação Wiki", "Transcrições de Cliente"],
        help="Escolha qual tipo de documento pesquisar"
    )

with col2:
    k_results = st.slider("Número de resultados:", min_value=1, max_value=10, value=5)

# Converte filtro para formato interno
source_type_filter = None
if source_filter == "Documentação Wiki":
    source_type_filter = "wiki_documentation"
elif source_filter == "Transcrições de Cliente":
    source_type_filter = "customer_transcript"

# Botão de busca
if st.button("🔍 Buscar na Base de Conhecimento", type="primary"):
    if query.strip():
        with st.spinner("Buscando na base de conhecimento..."):
            # Busca por chunks relevantes
            relevant_chunks = search_knowledge_base(
                query, index, chunks_data, embedding_model, 
                k=k_results, source_type_filter=source_type_filter
            )
            
            if relevant_chunks:
                # Gera resposta
                with st.spinner("Gerando resposta..."):
                    answer = generate_answer(query, relevant_chunks)
                
                # Exibe resultados
                st.markdown("---")
                st.markdown("### 🤖 Resposta")
                st.write(answer)
                
                # Mostra fontes
                sources_info = format_sources(relevant_chunks)
                st.markdown(sources_info)
                
                # Mostra chunks brutos para debug
                with st.expander("🔍 Ver Chunks Recuperados"):
                    for i, chunk in enumerate(relevant_chunks):
                        st.markdown(f"**Chunk {i+1}** (de {chunk.get('source_filename', 'Desconhecido')})")
                        st.text(chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'])
                        st.markdown("---")
            else:
                st.warning("Nenhuma informação relevante encontrada. Tente reformular sua pergunta ou verificar diferentes tipos de fonte.")
    else:
        st.warning("Por favor, digite uma pergunta para buscar.")

# Rodapé
st.markdown("---")
st.markdown("""
**💡 Dicas:**
- Tente diferentes formulações se não obtiver os resultados esperados
- Use o filtro de tipo de fonte para focar em tipos específicos de documento
- Verifique os chunks recuperados para ver exatamente quais informações foram encontradas
""")
