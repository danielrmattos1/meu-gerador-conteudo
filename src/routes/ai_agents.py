import os
from flask import Blueprint, request, jsonify
import google.generativeai as genai

ai_agents_bp = Blueprint('ai_agents', __name__)

# --- CONFIGURAÇÃO DA API ---
# Certifique-se de que a sua GOOGLE_AI_API_KEY está configurada no Render
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
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Erro ao chamar a API Gemini: {e}")
            return f"Erro ao gerar conteúdo: {e}"

class TopicsAgent(BaseAgent):
    """
    Agente especialista em criar tópicos profundos e filosóficos
    baseado nas suas instruções.
    """
    def process(self, title):
        prompt = f"""
        **MISSÃO:** Você é um especialista na criação de tópicos transformadores para vídeos do YouTube sobre Umbanda e Espiritualidade Crítica. Sua tarefa é transformar o tema '{title}' em uma lista de tópicos e subtópicos com profundidade filosófica, ética e simbólica, oferecendo clareza prática e evitando clichês.

        **ESTILO:** A linguagem deve ser acolhedora, reflexiva e potente, unindo o fundamento da Umbanda com autonomia pessoal e psicologia.

        **ESTRUTURA OBRIGATÓRIA PARA CADA TEMA:**
        - **Título do tema:** Uma frase magnetizante que mistura curiosidade, filosofia e impacto ético.
        - **Subtópico 1:** Uma crença limitante ou um dogma a ser desmantelado.
        - **Subtópico 2:** Um conceito pouco conhecido ou distorcido (ex: Teoria, Efeito Comportamental).
        - **Subtópico 3:** Uma revelação contraintuitiva que transfere o poder para o médium.
        - **Subtópico 4:** Uma consequência prática (o erro comportamental) ou a revelação de um relato real.
        - **Subtópico 5:** Uma metáfora simbólica ou um chamado ao estudo (livros).

        Gere os tópicos para o tema: '{title}'.
        """
        return self.call_gemini(prompt)

class ScriptAgent(BaseAgent):
    """
    Agente especialista em criar roteiros no estilo "Professor Reflexivo"
    baseado nas suas instruções.
    """
    def process(self, topics):
        prompt = f"""
        **MISSÃO:** Você é um roteirista para um canal Dark de YouTube sobre Umbanda e Espiritualidade Crítica. Sua tarefa é escrever um roteiro de 15-20 minutos, em texto corrido para narração, baseado nos seguintes tópicos. O objetivo é monetizar com afiliação de livros da Amazon.

        **TÓPICOS A SEREM DESENVOLVIDOS:**
        {topics}

        **ESTILO E TOM OBRIGATÓRIOS:**
        - **Tom de Professor Reflexivo:** Calmo, pausado, introspectivo e filosófico.
        - **Linguagem:** Evite clichês e valide a dor do espectador antes de oferecer a cura.
        - **Ênfase:** Na autonomia da consciência, não em rituais complexos.
        - **VETO TOTAL (BLACKLIST):** Nunca use termos como "Vibe", "Vibrar Alto", "Você Consegue", "Pense Positivo". A linguagem é de Mestre Filosófico, não de Coach Motivacional.

        **ESTRUTURA DO ROTEIRO:**
        1. **Abertura Filosófica:** Comece com uma pergunta profunda que desafie um dogma.
        2. **Desenvolvimento:** Explique os tópicos fornecidos, conectando a prática com a filosofia da autonomia. Integre sutilmente referências a estudos ou teorias.
        3. **Conclusão e CTA de Monetização:** Encerre de forma inspiradora e use a frase exata: "Se o seu poder está na sua consciência, o seu próximo passo está na sua lista de estudos. Encontre o Livro Essencial de Umbanda e todas as referências para sua jornada na descrição."
        
        **AJUSTE FINAL E RESTRIÇÃO CRÍTICA:**
        O resultado final deve ser um texto contínuo, limpo e sem interrupções. NÃO inclua absolutamente NENHUM tipo de formatação, como asteriscos para negrito (`**texto**`), hashtags para títulos (`# Título`), marcadores (`*`, `-`), ou qualquer outra sintaxe de markdown. O texto será inserido diretamente em um software de geração de áudio (Text-to-Speech) e qualquer formatação irá atrapalhar a narração. Entregue apenas o texto puro que será narrado.

        Agora, escreva o roteiro completo.
        """
        return self.call_gemini(prompt)

class ImagePromptAgent(BaseAgent):
    """
    Agente especialista em criar prompts para Midjourney no estilo
    "Neo-Ancestralismo Meditativo".
    """
    def process(self, script):
        # Usamos apenas os primeiros 2500 caracteres do roteiro para economizar tokens e focar na essência
        script_snippet = script[:2500]
        
        # AJUSTE: Modificado para gerar apenas 1 prompt de imagem
        prompt = f"""
        **MISSÃO:** Você é um especialista em engenharia de prompts para Midjourney. Sua tarefa é ler o trecho do roteiro de vídeo fornecido e criar 1 (um) único prompt visual, o mais poderoso e simbolicamente denso possível, que capture a essência da mensagem principal.

        **TRECHO DO ROTEIRO PARA ANÁLISE:**
        {script_snippet}

        **ESTILO VISUAL OBRIGATÓRIO: Neo-Ancestralismo Meditativo**
        - **Estética:** Pintura a óleo escura e dramática, textura de gravura, arte digital com textura de tela (Painterly).
        - **Temas:** Foco em metáforas visuais da autonomia e quebra de dogmas (ex: rochas rachadas com ouro kintsugi, chaves antigas, livros abertos na escuridão, raízes como nervos).
        - **Cores:** Fundos escuros (Preto Carvão, Azul-Marinho) com destaques em Ouro Velho, Cobre ou luz branca pura. Alto contraste.
        - **Composição:** Variada. Use close-ups, top view, silhuetas contemplativas em contraluz.
        - **Animação Sutil:** Inclua elementos com movimento lento, como "fumaça de incenso subindo lentamente" ou "chama da vela tremulando suavemente".

        **REGRAS:**
        - O prompt deve ser em inglês, detalhado e pronto para o Midjourney, no formato:
        `/imagine prompt: [descrição detalhada], [estilo], [parâmetros como --ar 16:9]`

        Crie 1 (um) único prompt baseado no roteiro.
        """
        return self.call_gemini(prompt)

# --- ROTA DA API ---
@ai_agents_bp.route('/process', methods=['POST'])
def process_title():
    data = request.get_json()
    if not data or 'title' not in data:
        return jsonify({'error': 'Título não fornecido'}), 400

    title = data['title']

    # Inicializa os agentes
    topics_agent = TopicsAgent(model)
    script_agent = ScriptAgent(model)
    image_prompt_agent = ImagePromptAgent(model)

    # Executa o pipeline
    print(f"Gerando tópicos para: {title}")
    topics = topics_agent.process(title)
    
    print("Gerando roteiro...")
    script = script_agent.process(topics)
    
    print("Gerando prompts de imagem...")
    image_prompts = image_prompt_agent.process(script)

    # Retorna os resultados
    return jsonify({
        'topics': topics,
        'script': script,
        'image_prompts': image_prompts
    })
