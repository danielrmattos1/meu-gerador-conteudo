import os
from flask import Blueprint, request, jsonify
import google.generativeai as genai

ai_agents_bp = Blueprint('ai_agents', __name__)

# --- CONFIGURAÇÃO DA API ---
api_key = os.getenv('GOOGLE_AI_API_KEY')
if not api_key:
    raise ValueError("API Key do Google não encontrada. Verifique as variáveis de ambiente.")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- DEFINIÇÃO DOS AGENTES ESPECIALISTAS ---

class BaseAgent:
    def __init__(self, model):
        self.model = model

    def process(self, data):
        raise NotImplementedError("O método process() deve ser implementado pela subclasse.")

    def call_gemini(self, prompt):
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        try:
            response = self.model.generate_content(prompt, safety_settings=safety_settings)
            # VERSÃO FINAL: Verifica se a resposta tem o atributo 'text' antes de retornar.
            if hasattr(response, 'text'):
                return response.text
            else:
                # Se não tiver 'text', a resposta foi bloqueada ou veio vazia.
                print(f"Resposta da API bloqueada ou vazia. Feedback: {response.prompt_feedback}")
                return "A resposta da IA foi bloqueada por filtros de segurança ou retornou vazia."
        except Exception as e:
            print(f"Erro ao chamar a API Gemini: {e}")
            return f"Erro ao gerar conteúdo: {e}"

class TopicsAgent(BaseAgent):
    def process(self, title):
        prompt = f"""
        **MISSÃO:** Você é um especialista na criação de tópicos transformadores para vídeos do YouTube sobre Umbanda e Espiritualidade Crítica. Sua tarefa é transformar o tema '{title}' em uma lista de tópicos e subtópicos com profundidade filosófica, ética e simbólica, oferecendo clareza prática e evitando clichês.
        **ESTILO:** A linguagem deve ser acolhedora, reflexiva e potente, unindo o fundamento da Umbanda com autonomia pessoal e psicologia.
        **ESTRUTURA OBRIGATÓRIA PARA CADA TEMA:**
        - **Título do tema:** Uma frase magnetizante.
        - **Subtópico 1:** Uma crença limitante a ser desmantelada.
        - **Subtópico 2:** Um conceito pouco conhecido ou distorcido.
        - **Subtópico 3:** Uma revelação contraintuitiva.
        - **Subtópico 4:** Uma consequência prática ou relato real.
        - **Subtópico 5:** Uma metáfora simbólica.
        Gere os tópicos para o tema: '{title}'.
        """
        return self.call_gemini(prompt)

class ScriptAgent(BaseAgent):
    def process(self, topics):
        prompt = f"""
        **MISSÃO:** Você é um roteirista para um canal Dark de YouTube sobre Umbanda e Espiritualidade Crítica. Escreva um roteiro de 15-20 minutos, em texto corrido para narração, baseado nos tópicos: {topics}.
        **ESTILO E TOM:** Tom de Professor Reflexivo: Calmo, pausado, introspectivo. Linguagem filosófica, sem clichês. Valide a dor do espectador. Ênfase na autonomia, não em rituais.
        **VETO TOTAL (BLACKLIST):** Nunca use "Vibe", "Vibrar Alto", "Você Consegue", "Pense Positivo". A linguagem é de Mestre Filosófico, não Coach.
        **ESTRUTURA:** 1. Abertura Filosófica. 2. Desenvolvimento dos tópicos. 3. Conclusão e CTA de Monetização com a frase exata: "Se o seu poder está na sua consciência, o seu próximo passo está na sua lista de estudos. Encontre o Livro Essencial de Umbanda e todas as referências para sua jornada na descrição."
        **RESTRIÇÃO CRÍTICA:** O resultado final deve ser um texto contínuo e limpo, SEM NENHUMA formatação (sem `**`, `#`, `*`, `-`). O texto será usado em um software de narração. Entregue apenas o texto puro a ser narrado.
        Agora, escreva o roteiro completo.
        """
        return self.call_gemini(prompt)

class ImagePromptAgent(BaseAgent):
    def process(self, script):
        script_snippet = script[:2500]
        prompt = f"""
        **MISSÃO:** Você é um especialista em prompts para Midjourney. Baseado no roteiro "{script_snippet}", crie 1 único prompt, o mais poderoso possível.
        **ESTILO VISUAL OBRIGATÓRIO: Neo-Ancestralismo Meditativo**
        - **Estética:** Pintura a óleo escura e dramática, textura de gravura, Painterly.
        - **Temas:** Metáforas visuais da autonomia (rochas rachadas com ouro kintsugi, chaves antigas, raízes como nervos).
        - **Cores:** Fundos escuros (Preto Carvão, Azul-Marinho) com destaques em Ouro Velho ou luz. Alto contraste.
        - **Composição:** Variada (close-ups, top view, silhuetas em contraluz).
        - **Animação Sutil:** Inclua elementos com movimento lento ("fumaça de incenso subindo lentamente").
        **REGRAS:** O prompt deve ser em inglês, detalhado, e no formato: `/imagine prompt: [descrição], [estilo], --ar 16:9`
        Crie 1 único prompt.
        """
        return self.call_gemini(prompt)

# --- ROTA DA API ---
@ai_agents_bp.route('/process', methods=['POST'])
def process_title():
    data = request.get_json()
    if not data or 'title' not in data:
        return jsonify({'error': 'Título não fornecido'}), 400
    title = data['title']
    topics_agent = TopicsAgent(model)
    script_agent = ScriptAgent(model)
    image_prompt_agent = ImagePromptAgent(model)
    topics = topics_agent.process(title)
    script = script_agent.process(topics)
    image_prompts = image_prompt_agent.process(script)
    return jsonify({
        'topics': topics,
        'script': script,
        'image_prompts': image_prompts
    })
