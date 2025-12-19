"""
Flask API Server für G-Retriever Obsidian Plugin
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from gretriever_inference import GraphRetriever

app = Flask(__name__)
CORS(app)  # Enable CORS for Obsidian

# Initialize retriever once at startup
print("Initializing G-Retriever...")
retriever = GraphRetriever(
    graph_path="./graph_output/graph.gpickle",
    ollama_model="llama3:8b",
    ollama_url="http://localhost:11434"
)
print("G-Retriever ready!")

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    question = data.get('question', '')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        result = retriever.query(question)
        return jsonify(result)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("G-Retriever API Server")
    print("Running on http://localhost:5000")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5001, debug=False)