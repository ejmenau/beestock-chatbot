# app.py
import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import google.generativeai as genai
import os
import re

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="KB BeeStock Chatbot",
    page_icon="🐝",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- FUNÇÃO DE CACHE PARA CARREGAR OS MODELOS E OS DADOS ---
@st.cache_resource
def load_rag_system():
    """
    Carrega todos os componentes necessários para o sistema RAG.
    """
    try:
        index = faiss.read_index("vector_index.faiss")
        with open("chunks_data.json", 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        bi_encoder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        return index, chunks_data, bi_encoder, cross_encoder
    except FileNotFoundError:
        st.error("Arquivos de índice (vector_index.faiss ou chunks_data.json) não encontrados. "
                 "Certifique-se de que eles estão no seu repositório GitHub junto com o app.py.")
        return None, None, None, None

# --- CARREGAMENTO DOS COMPONENTES DO SISTEMA ---
index, chunks_data, bi_encoder, cross_encoder = load_rag_system()

# --- CONFIGURAÇÃO DA API KEY DO GEMINI ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    st.sidebar.warning("🔑 Chave de API do Gemini não encontrada. Por favor, insira abaixo.")
    api_key = st.sidebar.text_input("Insira sua Chave de API do Gemini:", type="password", help="Obtenha sua chave em [Google AI Studio](https://makersuite.google.com/)")

if api_key:
    try:
        genai.configure(api_key=api_key)
        generative_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    except Exception as e:
        st.error(f"Erro ao configurar a API do Gemini: {e}")
        generative_model = None
else:
    generative_model = None

# --- FUNÇÕES DO CHATBOT ---

def search_knowledge_base(query, k=10):
    """
    Busca na base de conhecimento usando um processo de duas etapas: busca vetorial e re-ranking.
    """
    if bi_encoder is None or cross_encoder is None or index is None:
        return []

    # Etapa 1: Busca Vetorial Rápida em toda a base
    query_embedding = bi_encoder.encode([query])
    faiss.normalize_L2(query_embedding)
    
    # Busca um grande conjunto inicial de candidatos de toda a base de conhecimento
    distances, indices = index.search(query_embedding, 50) 
    candidate_chunks = [chunks_data[i] for i in indices[0] if i != -1]

    if not candidate_chunks:
        return []

    # Etapa 2: Re-ranking com Cross-Encoder
    cross_inp = [[query, chunk.get('text_chunk', '')] for chunk in candidate_chunks]
    cross_scores = cross_encoder.predict(cross_inp)

    for i in range(len(cross_scores)):
        candidate_chunks[i]['rerank_score'] = cross_scores[i]

    reranked_chunks = sorted(candidate_chunks, key=lambda x: x['rerank_score'], reverse=True)
    
    return reranked_chunks[:k]

def generate_response(query, retrieved_chunks):
    """
    Gera uma resposta usando o modelo Gemini com um prompt de "Chain of Thought".
    """
    if not retrieved_chunks or generative_model is None:
        return "Não foi possível encontrar informações relevantes para responder à sua pergunta.", []

    context_str = "\n\n---\n\n".join([f"Fonte: {chunk['metadata']['source_file']}\nConteúdo: {chunk.get('text_chunk', '')}" for chunk in retrieved_chunks])

    # O prompt "Chain of Thought" que força o raciocínio passo a passo
    prompt_template = f"""
    Siga estritamente este processo de raciocínio de 3 etapas para responder à pergunta do usuário.

    **Pergunta do Usuário:** "{query}"

    **Etapa 1: Identificar a Intenção Principal**
    Primeiro, identifique a intenção principal da pergunta do usuário. A pergunta é sobre um processo de cliente específico, uma funcionalidade geral do sistema (como fazer), ou está pedindo uma lista?

    **Etapa 2: Extrair Fatos Relevantes do Contexto**
    Segundo, revise o contexto fornecido abaixo e extraia textualmente todas as frases ou fatos que são diretamente relevantes para a intenção principal da pergunta. Liste cada fato. Se um fato menciona um nome de cliente ou pessoa, inclua esse nome.

    **Contexto Fornecido:**
    ---
    {context_str}
    ---

    **Etapa 3: Sintetizar a Resposta Final**
    Terceiro, com base nos fatos que você extraiu na Etapa 2, sintetize uma resposta final, abrangente e confiante em português do Brasil.
    - Se a pergunta for sobre "qual cliente" ou "quem", sua resposta DEVE nomear o cliente ou pessoa se essa informação estiver nos fatos extraídos.
    - Se a pergunta for sobre "como fazer", sua resposta DEVE fornecer um guia passo a passo, se os passos estiverem nos fatos extraídos.
    - Se, após extrair os fatos, a informação necessária para responder completamente não estiver lá, afirme que o contexto não fornece os detalhes específicos.

    Execute as três etapas internamente e forneça apenas a **Resposta Final Sintetizada** para o usuário.
    """
    
    try:
        response = generative_model.generate_content(prompt_template)
        return response.text, retrieved_chunks
    except Exception as e:
        return f"Ocorreu um erro ao gerar a resposta: {e}", []

# --- INTERFACE DO USUÁRIO (UI) ---
st.title("🐝 Chatbot da Base de Conhecimento BeeStock")
st.markdown("Faça uma pergunta sobre os processos dos clientes e o chatbot buscará a resposta nos documentos.")

# Sidebar
st.sidebar.header("Filtros de Busca (Opcional)")
if chunks_data:
    customer_list = sorted(list(set(chunk['metadata'].get('customer', 'N/A') for chunk in chunks_data)))
else:
    customer_list = []

# Inicializa o histórico do chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe as mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("Ver fontes utilizadas"):
                for source in message["sources"]:
                    st.info(f"**Fonte:** `{source['metadata']['source_file']}`\n\n---\n\n{source.get('text_chunk', '')}")

# Campo de entrada do usuário
if prompt := st.chat_input("Qual é a sua dúvida?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gera e exibe a resposta do assistente
    with st.chat_message("assistant"):
        prompt_lower = prompt.lower()
        if any(keyword in prompt_lower for keyword in ["exemplos de clientes", "quais clientes", "liste os clientes", "que clientes"]):
            # Filtra clientes "N/A" ou outros valores inválidos se existirem
            valid_customers = [c for c in customer_list if c != 'N/A' and c]
            if valid_customers:
                response = f"Com base nos documentos carregados, os clientes que usam o BeeStock são: **{', '.join(valid_customers)}**."
            else:
                response = "Ainda não tenho informações sobre clientes específicos na minha base de dados."
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response, "sources": []})
        
        else:
            with st.spinner("Analisando a base de conhecimento... 🧠"):
                if index is not None and generative_model is not None:
                    chunks = search_knowledge_base(prompt)
                    response, sources = generate_response(prompt, chunks)
                    
                    st.markdown(response)
                    
                    if sources:
                        with st.expander("Ver fontes utilizadas"):
                            for source in sources:
                                st.info(f"**Fonte:** `{source['metadata']['source_file']}`\n\n---\n\n{source.get('text_chunk', '')}")
                    
                    st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})
                elif index is None:
                     st.error("O sistema de busca não foi carregado.")
                else:
                     st.error("A chave de API do Gemini não foi configurada. Por favor, adicione-a.")

