import os
import json
import requests
from flask import Blueprint, request, jsonify
from flask_cors import cross_origin

ai_agents_bp = Blueprint('ai_agents', __name__)

class GoogleAIStudioAgent:
    """Classe base para interagir com agentes do Google AI Studio"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.headers = {
            "Content-Type": "application/json"
        }
    
    def call_gemini(self, prompt: str, model: str = "gemini-1.5-flash") -> str:
        """Faz uma chamada para a API do Gemini"""
        url = f"{self.base_url}/{model}:generateContent"
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        params = {"key": self.api_key}
        
        try:
            response = requests.post(url, json=payload, params=params, headers=self.headers)
            response.raise_for_status()
            
            result = response.json()
            if "candidates" in result and len(result["candidates"]) > 0:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                raise Exception("Resposta inválida da API")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Erro na chamada da API: {e}")

class TopicCreatorAgent(GoogleAIStudioAgent):
    """Agente para criar tópicos e subtópicos a partir de um título"""
    
    def create_topics(self, title: str) -> str:
        """Cria tópicos e subtópicos baseados no título fornecido"""
        prompt = f"""
        Com base no título "{title}", crie uma lista estruturada de tópicos e subtópicos para um conteúdo completo e abrangente.
        
        Formato de saída:
        - Tópico Principal 1
          - Subtópico 1.1
          - Subtópico 1.2
        - Tópico Principal 2
          - Subtópico 2.1
          - Subtópico 2.2
        
        Seja específico e organize os tópicos de forma lógica e sequencial.
        """
        
        return self.call_gemini(prompt)

class ScriptWriterAgent(GoogleAIStudioAgent):
    """Agente para criar roteiros baseados em tópicos"""
    
    def create_script(self, topics: str) -> str:
        """Cria um roteiro em texto corrido baseado nos tópicos fornecidos"""
        prompt = f"""
        Com base nos seguintes tópicos e subtópicos:
        
        {topics}
        
        Crie um roteiro detalhado em texto corrido. O roteiro deve:
        - Ser fluido e natural
        - Cobrir todos os tópicos mencionados
        - Ter uma introdução, desenvolvimento e conclusão
        - Ser adequado para apresentação ou narração
        - Ter aproximadamente 500-800 palavras
        
        Escreva o roteiro completo:
        """
        
        return self.call_gemini(prompt)

class ImagePromptAgent(GoogleAIStudioAgent):
    """Agente para gerar prompts de imagens baseados em roteiros"""
    
    def create_image_prompts(self, script: str) -> str:
        """Cria prompts para geração de imagens baseados no roteiro"""
        prompt = f"""
        Com base no seguinte roteiro:
        
        {script}
        
        Crie uma série de prompts detalhados para geração de imagens que complementem o conteúdo. 
        
        Para cada prompt, inclua:
        - Descrição visual específica
        - Estilo artístico sugerido
        - Elementos visuais importantes
        - Atmosfera/mood desejado
        
        Formato de saída:
        PROMPT 1: [descrição detalhada]
        PROMPT 2: [descrição detalhada]
        PROMPT 3: [descrição detalhada]
        
        Crie entre 3-5 prompts que capturem os momentos ou conceitos mais importantes do roteiro.
        """
        
        return self.call_gemini(prompt)

@ai_agents_bp.route('/process', methods=['POST'])
@cross_origin()
def process_title():
    """Endpoint principal que processa um título através dos três agentes"""
    
    try:
        # Obter dados da requisição
        data = request.get_json()
        if not data or 'title' not in data:
            return jsonify({'error': 'Título é obrigatório'}), 400
        
        title = data['title'].strip()
        if not title:
            return jsonify({'error': 'Título não pode estar vazio'}), 400
        
        # Verificar se a API key está configurada
        api_key = os.getenv('GOOGLE_AI_API_KEY')
        if not api_key:
            return jsonify({'error': 'API key do Google AI não configurada'}), 500
        
        # Criar instâncias dos agentes
        topic_agent = TopicCreatorAgent(api_key)
        script_agent = ScriptWriterAgent(api_key)
        image_agent = ImagePromptAgent(api_key)
        
        # Executar pipeline
        result = {'title': title}
        
        # Etapa 1: Criar tópicos
        try:
            topics = topic_agent.create_topics(title)
            result['topics'] = topics
        except Exception as e:
            return jsonify({'error': f'Erro ao criar tópicos: {str(e)}'}), 500
        
        # Etapa 2: Criar roteiro
        try:
            script = script_agent.create_script(topics)
            result['script'] = script
        except Exception as e:
            return jsonify({'error': f'Erro ao criar roteiro: {str(e)}'}), 500
        
        # Etapa 3: Criar prompts de imagem
        try:
            image_prompts = image_agent.create_image_prompts(script)
            result['image_prompts'] = image_prompts
        except Exception as e:
            return jsonify({'error': f'Erro ao criar prompts de imagem: {str(e)}'}), 500
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        return jsonify({'error': f'Erro interno: {str(e)}'}), 500

@ai_agents_bp.route('/health', methods=['GET'])
@cross_origin()
def health_check():
    """Endpoint para verificar se o serviço está funcionando"""
    api_key = os.getenv('GOOGLE_AI_API_KEY')
    return jsonify({
        'status': 'ok',
        'api_configured': bool(api_key)
    })

