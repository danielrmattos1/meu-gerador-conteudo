#!/usr/bin/env python3
"""
Script para automatizar o fluxo de trabalho dos agentes de IA:
1. Agente de criação de tópicos (título -> tópicos e subtópicos)
2. Agente de criação de roteiros (tópicos -> roteiro em texto corrido)
3. Agente de geração de prompts para imagens (roteiro -> prompts para IA de imagens)
"""

import os
import sys
import json
import time
import requests
from typing import Dict, Any, Optional

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

class AIAgentsOrchestrator:
    """Orquestrador principal que coordena todos os agentes"""
    
    def __init__(self, api_key: str):
        self.topic_agent = TopicCreatorAgent(api_key)
        self.script_agent = ScriptWriterAgent(api_key)
        self.image_agent = ImagePromptAgent(api_key)
    
    def run_full_pipeline(self, title: str) -> Dict[str, str]:
        """Executa o pipeline completo: título -> tópicos -> roteiro -> prompts de imagem"""
        
        print(f"🚀 Iniciando pipeline para o título: '{title}'")
        
        # Etapa 1: Criar tópicos
        print("📝 Etapa 1: Criando tópicos e subtópicos...")
        topics = self.topic_agent.create_topics(title)
        print("✅ Tópicos criados com sucesso!")
        
        # Etapa 2: Criar roteiro
        print("🎬 Etapa 2: Criando roteiro...")
        script = self.script_agent.create_script(topics)
        print("✅ Roteiro criado com sucesso!")
        
        # Etapa 3: Criar prompts de imagem
        print("🎨 Etapa 3: Criando prompts para imagens...")
        image_prompts = self.image_agent.create_image_prompts(script)
        print("✅ Prompts de imagem criados com sucesso!")
        
        return {
            "title": title,
            "topics": topics,
            "script": script,
            "image_prompts": image_prompts
        }
    
    def save_results(self, results: Dict[str, str], output_dir: str = "output"):
        """Salva os resultados em arquivos separados"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Salvar cada resultado em um arquivo separado
        files_created = []
        
        # Arquivo de tópicos
        topics_file = os.path.join(output_dir, "topicos.md")
        with open(topics_file, "w", encoding="utf-8") as f:
            f.write(f"# Tópicos para: {results['title']}\n\n")
            f.write(results['topics'])
        files_created.append(topics_file)
        
        # Arquivo de roteiro
        script_file = os.path.join(output_dir, "roteiro.md")
        with open(script_file, "w", encoding="utf-8") as f:
            f.write(f"# Roteiro para: {results['title']}\n\n")
            f.write(results['script'])
        files_created.append(script_file)
        
        # Arquivo de prompts de imagem
        prompts_file = os.path.join(output_dir, "prompts_imagem.md")
        with open(prompts_file, "w", encoding="utf-8") as f:
            f.write(f"# Prompts de Imagem para: {results['title']}\n\n")
            f.write(results['image_prompts'])
        files_created.append(prompts_file)
        
        # Arquivo JSON com todos os resultados
        json_file = os.path.join(output_dir, "resultados_completos.json")
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        files_created.append(json_file)
        
        print(f"📁 Resultados salvos em: {output_dir}")
        for file in files_created:
            print(f"   - {file}")
        
        return files_created

def main():
    """Função principal do script"""
    
    # Verificar se a API key foi fornecida
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        print("❌ Erro: Variável de ambiente GOOGLE_AI_API_KEY não encontrada!")
        print("   Configure sua API key do Google AI Studio como variável de ambiente.")
        sys.exit(1)
    
    # Verificar se o título foi fornecido como argumento
    if len(sys.argv) < 2:
        print("❌ Erro: Título não fornecido!")
        print("   Uso: python ai_agents_automation.py 'Seu Título Aqui'")
        sys.exit(1)
    
    title = sys.argv[1]
    
    try:
        # Criar o orquestrador e executar o pipeline
        orchestrator = AIAgentsOrchestrator(api_key)
        results = orchestrator.run_full_pipeline(title)
        
        # Salvar os resultados
        files_created = orchestrator.save_results(results)
        
        print("\n🎉 Pipeline executado com sucesso!")
        print(f"📊 Resultados disponíveis em {len(files_created)} arquivos.")
        
    except Exception as e:
        print(f"❌ Erro durante a execução: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
