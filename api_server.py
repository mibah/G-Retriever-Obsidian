"""
Flask API Server für G-Retriever Obsidian Plugin
Auto-detects free port and saves it for plugin
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from gretriever_inference import GraphRetriever
import socket
import json
from pathlib import Path

app = Flask(__name__)
CORS(app)

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
    return jsonify({'status': 'ok', 'message': 'G-Retriever is running'})


def find_free_port(start_port=5000, max_tries=10):
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + max_tries):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('', port))
            sock.close()
            return port
        except OSError:
            continue
    raise RuntimeError(f"No free port found between {start_port} and {start_port + max_tries}")


def save_port_config(port):
    """Save port to config file for plugin to read"""
    config_path = Path.home() / ".g-retriever-config.json"
    config = {
        "port": port,
        "url": f"http://localhost:{port}"
    }
    with open(config_path, 'w') as f:
        json.dump(config, f)
    print(f"✓ Config saved to: {config_path}")


if __name__ == '__main__':
    # Find free port
    port = find_free_port(start_port=5000)

    # Save config for plugin
    save_port_config(port)

    print("\n" + "=" * 60)
    print("G-Retriever API Server")
    print(f"Running on http://localhost:{port}")
    print("=" * 60)
    print("\nPlugin will automatically connect to this port.")
    print("Keep this window open while using the plugin.\n")

    app.run(host='0.0.0.0', port=port, debug=False)