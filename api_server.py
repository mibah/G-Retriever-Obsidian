"""
Flask API Server für G-Retriever Obsidian Plugin
Returns detailed debug info like terminal output
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from gretriever_inference import GraphRetriever
import socket
import json
from pathlib import Path
import sys

app = Flask(__name__)
CORS(app)

# Initialize retriever once at startup
print("Initializing G-Retriever...")
print("This may take a minute on first run...")

try:
    retriever = GraphRetriever(
        graph_path="./graph_output/graph.gpickle",
        ollama_model="llama3:8b",
        ollama_url="http://localhost:11434",
        verbose=True  # Show all debug output in terminal
    )
    print("✓ G-Retriever ready!")
except Exception as e:
    print(f"❌ Failed to initialize G-Retriever: {e}")
    sys.exit(1)


@app.route('/query', methods=['POST'])
def query():
    """Handle query requests from Obsidian plugin"""
    try:
        data = request.get_json()
        question = data.get('question', '')

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        print(f"\n[Query] {question}")

        # Process query - this includes all the debug output in terminal
        result = retriever.query(question)

        # Add debug info that was printed to result dict
        # This is already included in the query() method output

        print(f"[Response] Generated answer with {len(result.get('subgraph_nodes', []))} source nodes")
        print(f"\nAntwort: {result['answer']}")
        print(f"\nVerwendete Notizen: {', '.join(result.get('subgraph_nodes', [])[:5])}")
        print(f"\n[Answer Preview] {result['answer'][:100]}...")

        # Return result with all debug info
        response_data = {
            "question": result["question"],
            "answer": result["answer"],
            "retrieved_nodes": result.get("retrieved_nodes", []),
            "subgraph_nodes": result.get("subgraph_nodes", []),
            "debug_info": {
                "retrieved_indices": result.get("retrieved_indices", []),
                "node_list_length": len(retriever.node_list),
                "retrieved_names": result.get("retrieved_names", [])
            }
        }

        return jsonify(response_data)

    except Exception as e:
        print(f"[Error] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'G-Retriever is running',
        'model': retriever.ollama_model,
        'nodes': len(retriever.node_list)
    })


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
    try:
        port = find_free_port(start_port=5000)
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # Save config for plugin
    save_port_config(port)

    print("\n" + "=" * 60)
    print("G-Retriever API Server")
    print(f"Running on http://localhost:{port}")
    print("=" * 60)
    print("\n✓ Plugin will automatically connect to this port.")
    print("✓ Keep this window open while using the plugin.")
    print("\nEndpoints:")
    print(f"  - POST http://localhost:{port}/query")
    print(f"  - GET  http://localhost:{port}/health")
    print("\nPress Ctrl+C to stop the server\n")

    try:
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped gracefully")
        sys.exit(0)