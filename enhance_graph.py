"""
Graph Enhancement: Verbessere Connectivity für Low-Degree Nodes
"""

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


def enhance_graph_connectivity(graph_path: str, output_path: str,
                               min_degree: int = 2,
                               max_new_edges_per_node: int = 3):
    """
    Verbindet Low-Degree Nodes mit semantisch ähnlichen Nodes
    """

    # Load graph
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)

    print(f"Original graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")

    # Find low-degree nodes
    low_degree = [(n, d) for n, d in graph.degree() if d < min_degree]
    print(f"\nFound {len(low_degree)} nodes with degree < {min_degree}")

    if not low_degree:
        print("No enhancement needed!")
        return

    # Compute embeddings for semantic similarity
    print("\nComputing semantic similarities...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    node_list = list(graph.nodes())
    node_contents = []

    for node in node_list:
        content = graph.nodes[node].get('content', '')
        title = node
        text = f"{title}. {content[:500]}"
        node_contents.append(text)

    embeddings = embedder.encode(node_contents, show_progress_bar=True)

    # For each low-degree node, find similar nodes
    edges_added = 0

    for node, degree in low_degree:
        node_idx = node_list.index(node)
        node_emb = embeddings[node_idx].reshape(1, -1)

        # Compute similarities
        similarities = cosine_similarity(node_emb, embeddings)[0]

        # Exclude self and existing neighbors
        neighbors = set(graph.neighbors(node))
        similarities[node_idx] = -1  # Exclude self
        for neighbor in neighbors:
            neighbor_idx = node_list.index(neighbor)
            similarities[neighbor_idx] = -1  # Exclude existing

        # Find top-k most similar
        k = max_new_edges_per_node - degree  # Fill up to min_degree
        if k > 0:
            top_k_indices = np.argsort(similarities)[-k:][::-1]

            for idx in top_k_indices:
                if similarities[idx] > 0.3:  # Threshold
                    target_node = node_list[idx]
                    graph.add_edge(node, target_node)
                    edges_added += 1
                    print(f"  Added edge: {node} -> {target_node} (sim: {similarities[idx]:.3f})")

    print(f"\nAdded {edges_added} new edges")
    print(f"Enhanced graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")

    # Save enhanced graph
    with open(output_path, 'wb') as f:
        pickle.dump(graph, f)

    print(f"Saved enhanced graph to: {output_path}")

    # New statistics
    degrees = [d for n, d in graph.degree()]
    print(f"\nNew statistics:")
    print(f"  Average degree: {np.mean(degrees):.2f}")
    print(f"  Nodes with degree < {min_degree}: {sum(1 for d in degrees if d < min_degree)}")


if __name__ == "__main__":
    enhance_graph_connectivity(
        graph_path="./graph_output/graph.gpickle",
        output_path="./graph_output/graph_enhanced.gpickle",
        min_degree=2,
        max_new_edges_per_node=3
    )