# app.py
import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="KB BeeStock Chatbot",
    page_icon="🐝",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- FUNÇÃO DE CACHE PARA CARREGAR O MODELO E OS DADOS ---
# O cache acelera o carregamento, executando esta função apenas uma vez.
@st.cache_resource
def load_rag_system():
    """
    Carrega todos os componentes necessários para o sistema RAG.
    Isso inclui o índice FAISS, os dados dos chunks e o modelo de embedding.
    """
    try:
        index = faiss.read_index("vector_index.faiss")
        with open("chunks_data.json", 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        # **FIX**: Using the powerful multilingual model
        embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        return index, chunks_data, embedding_model
    except FileNotFoundError:
        st.error("Arquivos de índice (vector_index.faiss ou chunks_data.json) não encontrados. "
                 "Por favor, execute o script `build_index.py` primeiro.")
        return None, None, None

# --- CARREGAMENTO DOS COMPONENTES DO SISTEMA ---
index, chunks_data, embedding_model = load_rag_system()

# --- CONFIGURAÇÃO DA API KEY DO GEMINI ---
# Use o segredo do Streamlit para armazenar a chave de API de forma segura
try:
    # Tenta obter a chave da API dos segredos do Streamlit (melhor para deploy)
    api_key = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    # Se não encontrar, pede ao usuário para inserir no sidebar (bom para desenvolvimento local)
    st.sidebar.warning("🔑 Chave de API do Gemini não encontrada. Por favor, insira abaixo.")
    api_key = st.sidebar.text_input("Insira sua Chave de API do Gemini:", type="password", help="Obtenha sua chave em [Google AI Studio](https://makersuite.google.com/)")

if api_key:
    try:
        genai.configure(api_key=api_key)
        # **FIX**: Using the latest stable model 'gemini-1.5-flash-latest'
        generative_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    except Exception as e:
        st.error(f"Erro ao configurar a API do Gemini: {e}")
        generative_model = None
else:
    generative_model = None


# --- FUNÇÕES DO CHATBOT ---
def search_knowledge_base(query, k=5, customer_filter=None):
    """
    Busca na base de conhecimento os chunks mais relevantes para a consulta.
    """
    if embedding_model is None or index is None:
        return []

    query_embedding = embedding_model.encode([query])
    faiss.normalize_L2(query_embedding)

    # A busca no FAISS retorna distâncias e índices
    distances, indices = index.search(query_embedding, k * 2) # Busca mais para garantir resultados após o filtro

    retrieved_chunks = []
    for i in indices[0]:
        if i != -1:
            chunk_info = chunks_data[i]
            # Aplica o filtro de cliente se fornecido
            if customer_filter:
                if "metadata" in chunk_info and chunk_info["metadata"].get("customer", "").lower() == customer_filter.lower():
                    retrieved_chunks.append(chunk_info)
            else:
                retrieved_chunks.append(chunk_info)
    
    return retrieved_chunks[:k] # Retorna o número correto de chunks após o filtro

def generate_response(query, retrieved_chunks):
    """
    Gera uma resposta usando o modelo Gemini com base nos chunks recuperados.
    """
    if not retrieved_chunks or generative_model is None:
        return "Não foi possível encontrar informações relevantes para responder à sua pergunta."

    context_str = "\n\n---\n\n".join([chunk["text_chunk"] for chunk in retrieved_chunks if "text_chunk" in chunk])

    prompt_template = f"""
    Você é um analista de suporte especialista no sistema WMS da BeeStock. Sua tarefa é responder à pergunta do usuário de forma clara e objetiva, em português do Brasil.
    
    Use SOMENTE o contexto fornecido abaixo para formular sua resposta. Não invente informações.
    Se a informação não estiver no contexto, diga que não possui informações suficientes para responder.

    CONTEXTO:
    {context_str}

    PERGUNTA:
    {query}

    RESPOSTA:
    """
    
    try:
        response = generative_model.generate_content(prompt_template)
        return response.text
    except Exception as e:
        return f"Ocorreu um erro ao gerar a resposta: {e}"

# --- INTERFACE DO USUÁRIO (UI) ---
st.title("🐝 Chatbot da Base de Conhecimento BeeStock")
st.markdown("Faça uma pergunta sobre os processos dos clientes e o chatbot buscará a resposta nos documentos.")

# Sidebar para filtros
st.sidebar.header("Filtros de Busca")
customer_list = sorted(list(set(chunk['metadata']['customer'] for chunk in chunks_data if 'metadata' in chunk and 'customer' in chunk['metadata']))) if chunks_data else []

customer_filter = st.sidebar.selectbox(
    "Filtrar por cliente (opcional):",
    options=["Todos"] + customer_list
)
customer_filter_value = None if customer_filter == "Todos" else customer_filter

# Inicializa o histórico do chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe as mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Campo de entrada do usuário
if prompt := st.chat_input("Qual é a sua dúvida?"):
    # Adiciona a mensagem do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gera e exibe a resposta do assistente
    with st.chat_message("assistant"):
        with st.spinner("Pensando... 🧠"):
            if index is not None and generative_model is not None:
                chunks = search_knowledge_base(prompt, customer_filter=customer_filter_value)
                response = generate_response(prompt, chunks)
                st.markdown(response)
                # Adiciona a resposta do assistente ao histórico
                st.session_state.messages.append({"role": "assistant", "content": response})
            elif index is None:
                 st.error("O sistema de busca não foi carregado. Verifique os arquivos de índice.")
            else:
                 st.error("A chave de API do Gemini não foi configurada. Por favor, adicione-a no menu lateral.")
