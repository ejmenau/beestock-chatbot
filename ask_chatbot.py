import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

class RAGSystem:
    """
    Sistema RAG (Recuperação e Geração Aumentada) para suporte ao BeeStock WMS.
    Carrega índice vetorial pré-construído e fornece capacidades de resposta a perguntas.
    """
    
    def __init__(self, index_file='vector_index.faiss', chunks_file='chunks_data.json'):
        """
        Inicializa o sistema RAG carregando o índice vetorial, dados dos chunks e modelo de embedding.
        
        Args:
            index_file (str): Caminho para o arquivo de índice FAISS
            chunks_file (str): Caminho para o arquivo JSON contendo chunks de texto e metadados
        """
        print("Inicializando Sistema RAG...")
        
        # Carrega o índice vetorial FAISS
        print("Carregando índice vetorial...")
        self.index = faiss.read_index(index_file)
        print(f"✅ Índice vetorial carregado: {self.index.ntotal} vetores, {self.index.d} dimensões")
        
        # Carrega os dados dos chunks
        print("Carregando dados dos chunks...")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            self.chunks_data = json.load(f)
        print(f"✅ Dados dos chunks carregados: {len(self.chunks_data)} chunks")
        
        # Carrega o modelo de embedding
        print("Carregando modelo de embedding...")
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        print("✅ Modelo de embedding carregado")
        
        # Analisa a base de conhecimento
        self._analyze_knowledge_base()
        
        print("Inicialização do Sistema RAG concluída! 🚀")
        print()
    
    def _analyze_knowledge_base(self):
        """Analisa a base de conhecimento carregada para mostrar o que está disponível."""
        source_types = {}
        customer_names = set()
        
        for chunk in self.chunks_data:
            source_type = chunk.get('source_type', 'unknown')
            source_types[source_type] = source_types.get(source_type, 0) + 1
            
            if chunk.get('customer_name') != 'Documentação Wiki':
                customer_names.add(chunk.get('customer_name', 'Desconhecido'))
        
        print("📊 Análise da Base de Conhecimento:")
        for source_type, count in source_types.items():
            if source_type == 'wiki_documentation':
                print(f"  - Documentação Wiki: {count} chunks")
            elif source_type == 'customer_transcript':
                print(f"  - Transcrições de Cliente: {count} chunks")
            else:
                print(f"  - {source_type}: {count} chunks")
        
        if customer_names:
            print(f"  - Clientes: {', '.join(sorted(customer_names))}")
        print()
    
    def search(self, query, k=5, customer_filter=None, source_type_filter=None):
        """
        Busca pelos chunks de texto mais relevantes baseado na consulta do usuário.
        
        Args:
            query (str): A pergunta do usuário
            k (int): Número de resultados principais para retornar (padrão: 5)
            customer_filter (str, opcional): Filtra resultados por nome específico do cliente
            source_type_filter (str, opcional): Filtra por tipo de fonte ('customer_transcript' ou 'wiki_documentation')
            
        Returns:
            list: Lista de chunks relevantes com metadados
        """
        # Gera embedding para a consulta
        query_embedding = self.model.encode([query])
        
        # Normaliza o embedding da consulta para similaridade de cosseno
        faiss.normalize_L2(query_embedding)
        
        # Busca no índice FAISS com mais resultados para permitir filtragem
        search_k = min(k * 3, len(self.chunks_data))  # Obtém mais resultados para filtrar
        distances, indices = self.index.search(query_embedding, search_k)
        
        # Recupera os chunks correspondentes
        relevant_chunks = []
        for idx in indices[0]:
            if idx < len(self.chunks_data):  # Garante que o índice é válido
                chunk = self.chunks_data[idx]
                
                # Aplica filtro de cliente se especificado
                if customer_filter and chunk.get('customer_name') != 'Documentação Wiki':
                    if chunk['customer_name'].lower() != customer_filter.lower():
                        continue
                
                # Aplica filtro de tipo de fonte se especificado
                if source_type_filter:
                    if chunk.get('source_type') != source_type_filter:
                        continue
                
                relevant_chunks.append(chunk)
                
                # Para quando temos resultados suficientes
                if len(relevant_chunks) >= k:
                    break
        
        return relevant_chunks
    
    def ask(self, query, customer_filter=None, source_type_filter=None):
        """
        Método RAG principal: recupera contexto relevante e gera uma resposta usando Gemini.
        
        Args:
            query (str): A pergunta do usuário
            customer_filter (str, opcional): Filtra resultados por nome específico do cliente
            source_type_filter (str, opcional): Filtra por tipo de fonte
            
        Returns:
            str: Resposta gerada do Gemini baseada no contexto recuperado
        """
        # Busca por chunks de contexto relevantes
        relevant_chunks = self.search(query, k=5, customer_filter=customer_filter, source_type_filter=source_type_filter)
        
        if not relevant_chunks:
            return "Não consegui encontrar informações relevantes na base de conhecimento para responder sua pergunta. Por favor, tente reformular sua pergunta ou perguntar sobre um tópico diferente."
        
        # Combina os chunks recuperados em contexto
        context_str = "\n\n---\n".join([chunk['text'] for chunk in relevant_chunks])
        
        # Cria template de prompt detalhado para Gemini
        prompt_template = f"""Você é um analista de suporte especialista no sistema WMS BeeStock (Sistema de Gerenciamento de Armazém). 
Seu papel é ajudar os usuários a entender processos, procedimentos e informações relacionadas ao BeeStock.

IMPORTANTE: Responda à pergunta do usuário baseado APENAS nas informações fornecidas no contexto abaixo. 
Não use conhecimento externo ou faça suposições além do que está declarado no contexto.
Se o contexto não contiver informações suficientes para responder completamente à pergunta, diga isso claramente.

Informações do contexto:
{context_str}

Pergunta do usuário: {query}

Por favor, forneça uma resposta clara e útil baseada no contexto acima:"""

        try:
            # Gera resposta usando Gemini Flash (melhor para free tier)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt_template)
            
            # Extrai a resposta de texto
            answer = response.text.strip()
            
            # Adiciona informações detalhadas das fontes
            sources_info = self._format_sources(relevant_chunks)
            
            return answer + sources_info
            
        except Exception as e:
            return f"Desculpe, encontrei um erro ao gerar a resposta: {str(e)}. Por favor, tente novamente."
    
    def _format_sources(self, chunks):
        """Formata informações das fontes para a resposta."""
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

def main():
    """Função de execução principal para o chatbot RAG."""
    
    # IMPORTANTE: Substitua 'YOUR_GEMINI_API_KEY' pela sua chave de API real do Gemini
    # Você pode obter uma em: https://makersuite.google.com/app/apikey
    genai.configure(api_key="AIzaSyApjah1pIDAbpG7O2guSi1UcqFKwrEo7hs")
    
    print("=" * 70)
    print("🤖 Chatbot RAG BeeStock WMS")
    print("=" * 70)
    print("Este chatbot pode responder perguntas sobre processos do BeeStock WMS")
    print("baseado na base de conhecimento de documentação de clientes e artigos wiki.")
    print()
    
    try:
        # Inicializa o sistema RAG
        rag_system = RAGSystem()
        
        print("Sessão de chat iniciada! Digite 'sair' para sair.")
        print("-" * 50)
        
        # Loop principal de chat
        while True:
            try:
                # Obtém entrada do usuário
                user_query = input("\n💬 Faça sua pergunta (ou digite 'sair' para sair): ").strip()
                
                # Verifica comando de saída
                if user_query.lower() in ['sair', 'quit', 'bye', 'exit']:
                    print("\n👋 Obrigado por usar o Chatbot RAG BeeStock WMS!")
                    break
                
                # Pula consultas vazias
                if not user_query:
                    print("Por favor, digite uma pergunta.")
                    continue
                
                # Pergunta sobre preferências de busca
                print("\n🔍 Opções de Busca:")
                print("1. Buscar tudo (padrão)")
                print("2. Buscar apenas documentação wiki")
                print("3. Buscar apenas transcrições de cliente")
                print("4. Buscar cliente específico")
                
                choice = input("Escolha a opção (1-4, padrão: 1): ").strip()
                
                customer_filter = None
                source_type_filter = None
                
                if choice == "2":
                    source_type_filter = "wiki_documentation"
                    print("🔍 Buscando apenas documentação wiki...")
                elif choice == "3":
                    source_type_filter = "customer_transcript"
                    print("🔍 Buscando apenas transcrições de cliente...")
                elif choice == "4":
                    customer_filter = input("Digite o nome do cliente: ").strip()
                    if customer_filter:
                        print(f"🔍 Filtrando resultados para o cliente: {customer_filter}")
                else:
                    print("🔍 Buscando todas as fontes...")
                
                print("\n🔍 Buscando na base de conhecimento...")
                
                # Obtém resposta do sistema RAG
                answer = rag_system.ask(user_query, customer_filter, source_type_filter)
                
                # Exibe a resposta
                print("\n" + "=" * 50)
                print("🤖 RESPOSTA:")
                print("=" * 50)
                print(answer)
                print("=" * 50)
                
            except KeyboardInterrupt:
                print("\n\n👋 Sessão de chat interrompida. Tchau!")
                break
            except Exception as e:
                print(f"\n❌ Erro: {str(e)}")
                print("Por favor, tente novamente ou digite 'sair' para sair.")
    
    except FileNotFoundError as e:
        print(f"❌ Erro: Não foi possível encontrar os arquivos necessários: {str(e)}")
        print("Por favor, certifique-se de que você executou 'build_index.py' primeiro para criar os arquivos de índice.")
    except Exception as e:
        print(f"❌ Erro fatal: {str(e)}")
        print("Por favor, verifique sua configuração e tente novamente.")

if __name__ == "__main__":
    main()
