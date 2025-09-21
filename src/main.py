import os
from flask import Flask, send_from_directory
from src.routes.ai_agents import ai_agents_bp

# Cria a aplicação Flask, apontando para a pasta 'static' para os arquivos HTML/CSS/JS
app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))

# Registra apenas o blueprint dos agentes de IA
app.register_blueprint(ai_agents_bp, url_prefix='/api/ai')

# Rota principal para servir o arquivo index.html e outros arquivos estáticos
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder_path = app.static_folder
    if not static_folder_path:
        return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return "index.html not found", 404

# Bloco de inicialização para produção, compatível com o Render
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
