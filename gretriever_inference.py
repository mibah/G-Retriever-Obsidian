"""
G-Retriever Inference & Chat Interface
Nutzt PyTorch Geometric's G-Retriever für QA über den Obsidian Graph.
"""

import torch
import networkx as nx
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from pcst_fast import pcst_fast


class GraphRetriever:
    """
    Vereinfachter G-Retriever für Inference
    Nutzt: RAG (Retrieval) + GNN Encoding + Ollama LLM
    """

    def __init__(self,
                 graph_path: str,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 gnn_hidden: int = 256,
                 gnn_layers: int = 3,
                 ollama_model: str = "llama3:8b",
                 ollama_url: str = "http://localhost:11434",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 verbose: bool = True):

        self.device = device
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.verbose = verbose

        # Lade Graph
        import pickle

        self._log(f"Loading graph from {graph_path}...")
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)

        self.node_list = list(self.graph.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.node_list)}

        # Embedder
        self._log("Loading embedding model...")
        self.embedder = SentenceTransformer(embedding_model)

        # Pre-compute node embeddings
        self._log("Computing node embeddings...")
        self._compute_node_embeddings()

        # Build edge index
        self._build_edge_index()

        # GNN Model (simple GAT)
        embed_dim = self.node_embeddings.shape[1]
        self.gnn = SimpleGNN(embed_dim, gnn_hidden, gnn_layers).to(device)

        self._log("GraphRetriever ready!")

    def _log(self, message: str):
        """Internal logging - only prints if verbose=True"""
        if self.verbose:
            print(message)

    def _compute_node_embeddings(self):
        """Pre-compute embeddings für alle Knoten"""
        node_texts = []
        for node in self.node_list:
            content = self.graph.nodes[node].get("content", "")
            text = f"{node}. {content[:500]}"
            node_texts.append(text)

        self.node_embeddings = self.embedder.encode(
            node_texts,
            batch_size=32,
            show_progress_bar=self.verbose,
            convert_to_tensor=True,
            device=self.device
        )

    def _build_edge_index(self):
        """Erstellt Edge Index"""
        edge_list = list(self.graph.edges())
        if len(edge_list) == 0:
            # Fallback: vollständiger Graph für disconnected nodes
            self.edge_index = torch.tensor([[],[]], dtype=torch.long)
        else:
            edge_index = torch.tensor([
                [self.node_to_idx[u] for u, v in edge_list],
                [self.node_to_idx[v] for u, v in edge_list]
            ], dtype=torch.long)
            # Beide Richtungen für undirected
            self.edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    def retrieve_relevant_nodes(self, query: str, k: int = 10) -> List[int]:
        """
        Retrieval: Findet die k relevantesten Knoten für die Query
        """
        query_emb = self.embedder.encode(
            query,
            convert_to_tensor=True,
            device=self.device
        )

        # Cosine Similarity
        similarities = F.cosine_similarity(
            query_emb.unsqueeze(0),
            self.node_embeddings,
            dim=1
        )

        # Top-k
        top_k_indices = torch.topk(similarities, k=min(k, len(similarities))).indices
        return top_k_indices.cpu().tolist()

    def construct_subgraph_pcst(self, relevant_nodes: List[int],
                                query_emb: torch.Tensor) -> List[int]:
        """
        Subgraph Construction mit PCST (Prize-Collecting Steiner Tree)
        Findet optimal connected subgraph
        """
        if len(relevant_nodes) <= 3:
            return relevant_nodes

        try:
            # Prizes: Relevanz-Scores
            prizes = torch.zeros(len(self.node_list))
            similarities = F.cosine_similarity(
                query_emb.unsqueeze(0),
                self.node_embeddings,
                dim=1
            )
            prizes[relevant_nodes] = similarities[relevant_nodes].cpu()

            # Edges + Costs
            edges = self.edge_index.t().cpu().numpy()
            costs = np.ones(edges.shape[0])  # Uniform cost

            # PCST
            root = relevant_nodes[0]
            vertices, _ = pcst_fast(
                edges, prizes.numpy(), costs, root, 1, 1, 'strong'
            )

            return vertices.tolist()[:15]  # Max 15 Knoten
        except:
            # Fallback
            return relevant_nodes[:10]

    def encode_subgraph(self, subgraph_nodes: List[int]) -> torch.Tensor:
        """
        Encode subgraph mit GNN
        """
        # Subgraph Data
        x = self.node_embeddings[subgraph_nodes]

        # Subgraph edges (re-index)
        node_set = set(subgraph_nodes)
        old_to_new = {old: new for new, old in enumerate(subgraph_nodes)}

        subgraph_edges = []
        for i in range(self.edge_index.shape[1]):
            u, v = self.edge_index[:, i].tolist()
            if u in node_set and v in node_set:
                subgraph_edges.append([old_to_new[u], old_to_new[v]])

        if len(subgraph_edges) == 0:
            # No edges: return mean of node embeddings
            return x.mean(dim=0)

        edge_index = torch.tensor(subgraph_edges, dtype=torch.long).t()

        # GNN forward
        with torch.no_grad():
            graph_emb = self.gnn(x, edge_index)

        return graph_emb

    def generate_answer(self, query: str, subgraph_nodes: List[int]) -> str:
        """
        Generate answer mit Ollama LLM
        """
        import requests

        # Erstelle Context aus Subgraph
        context_parts = []
        for idx in subgraph_nodes[:10]:  # Max 10 nodes für Context
            # Prüfe ob Index gültig ist
            if idx < 0 or idx >= len(self.node_list):
                continue

            node = self.node_list[idx]

            # Prüfe ob Knoten wirklich im Graph existiert
            if node not in self.graph.nodes:
                continue

            content = self.graph.nodes[node].get("content", "")[:500]
            context_parts.append(f"**{node}**:\n{content}")

        context = "\n\n".join(context_parts)

        prompt = f"""Basierend auf den folgenden Notizen, beantworte die Frage präzise und informativ.

Notizen:
{context}

Frage: {query}

Antwort:"""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.5
                },
                timeout=120
            )
            response.raise_for_status()
            answer = response.json()["response"]
            return answer.strip()
        except Exception as e:
            return f"Error generating answer: {e}"

    def query(self, question: str, k_retrieve: int = 20) -> Dict:
        """
        Vollständiger RAG Pipeline: Retrieve -> Construct -> Generate
        """
        self._log(f"\nQuery: {question}")

        # 1. Retrieval
        self._log("Retrieving relevant nodes...")
        relevant_nodes = self.retrieve_relevant_nodes(question, k=k_retrieve)

        if self.verbose:
            self._log(f"DEBUG: Retrieved node indices: {relevant_nodes}")
            self._log(f"DEBUG: Node list length: {len(self.node_list)}")
            valid_nodes = [self.node_list[i] for i in relevant_nodes if i < len(self.node_list)]
            self._log(f"DEBUG: Retrieved node names: {valid_nodes}")

        # 2. Subgraph Construction (PCST)
        self._log("Constructing subgraph...")
        query_emb = self.embedder.encode(question, convert_to_tensor=True)
        subgraph_nodes = self.construct_subgraph_pcst(relevant_nodes, query_emb)

        # 3. Generate
        self._log("Generating answer...")
        answer = self.generate_answer(question, subgraph_nodes)

        # Return mit Metadaten und Debug-Infos
        retrieved_names = [self.node_list[i] for i in relevant_nodes if i < len(self.node_list)]
        return {
            "question": question,
            "answer": answer,
            "retrieved_nodes": [self.node_list[i] for i in relevant_nodes[:5] if i < len(self.node_list)],
            "subgraph_nodes": [self.node_list[i] for i in subgraph_nodes if i < len(self.node_list)],
            "retrieved_indices": relevant_nodes,
            "retrieved_names": retrieved_names
        }


class SimpleGNN(torch.nn.Module):
    """Einfaches GAT-basiertes GNN für Graph Encoding"""

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int):
        super().__init__()
        self.convs = torch.nn.ModuleList()

        self.convs.append(GATConv(in_channels, hidden_channels, heads=4, concat=True))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels * 4, hidden_channels, heads=4, concat=True))

        self.out_proj = torch.nn.Linear(hidden_channels * 4, hidden_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        # Global pooling
        graph_emb = x.mean(dim=0)
        return self.out_proj(graph_emb)


class InteractiveChatInterface:
    """Interaktives Chat Interface"""

    def __init__(self, retriever: GraphRetriever):
        self.retriever = retriever
        self.history = []

    def run(self):
        """Startet interaktiven Chat"""
        print("\n" + "="*60)
        print("G-Retriever Chat Interface für Obsidian Vault")
        print("Tippe 'quit' oder 'exit' zum Beenden")
        print("="*60 + "\n")

        while True:
            try:
                question = input("\nDeine Frage: ").strip()

                if question.lower() in ['quit', 'exit', 'q']:
                    print("Bis später!")
                    break

                if not question:
                    continue

                # Query
                result = self.retriever.query(question)

                # Output
                print(f"\nAntwort: {result['answer']}")
                print(f"\nVerwendete Notizen: {', '.join(result['subgraph_nodes'][:5])}")

                self.history.append(result)

            except KeyboardInterrupt:
                print("\n\nBis später!")
                break
            except Exception as e:
                print(f"\nFehler: {e}")


def main():
    """Direkte Nutzung ohne CLI"""

    graph_path = "./graph_output/graph.gpickle"
    ollama_model = "llama3:8b"
    ollama_url = "http://localhost:11434"
    embedding_model = "all-MiniLM-L6-v2"

    # Initialisiere Retriever
    retriever = GraphRetriever(
        graph_path,
        embedding_model=embedding_model,
        ollama_model=ollama_model,
        ollama_url=ollama_url,
        verbose=True  # Set to False for server usage
    )

    # Starte Chat
    chat = InteractiveChatInterface(retriever)
    chat.run()

if __name__ == "__main__":
    main()