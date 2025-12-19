"""
G-Retriever Inference mit TRAINIERTEM Model
"""

import torch
from gretriever_inference import GraphRetriever, InteractiveChatInterface
from train_gretriever import GRetrieverModel


class TrainedGraphRetriever(GraphRetriever):
    """GraphRetriever mit trainiertem GNN"""

    def __init__(self, graph_path: str, model_path: str, **kwargs):
        super().__init__(graph_path, **kwargs)

        # Lade trainiertes Model
        print(f"Loading trained model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)

        embed_dim = self.node_embeddings.shape[1]
        self.trained_model = GRetrieverModel(
            node_embed_dim=embed_dim,
            hidden_dim=256,
            num_layers=3,
            num_heads=4
        ).to(self.device)

        self.trained_model.load_state_dict(checkpoint['model_state_dict'])
        self.trained_model.eval()

        print("✓ Trained model loaded!")

    def retrieve_relevant_nodes(self, query: str, k: int = 10):
        """Retrieval mit trainiertem Model"""
        query_emb = self.embedder.encode(
            query,
            convert_to_tensor=True,
            device=self.device
        )

        # Nutze trainiertes Model für scoring
        with torch.no_grad():
            scores = self.trained_model(
                self.node_embeddings,
                self.edge_index.to(self.device),
                query_emb.unsqueeze(0)
            )

        # Top-k
        top_k_indices = torch.topk(scores, k=min(k, len(scores))).indices
        return top_k_indices.cpu().tolist()


def main():
    """Chat mit trainiertem Model"""

    graph_path = "./graph_output/graph.gpickle"
    model_path = "./models/best_model.pt"  # Oder final_model.pt

    retriever = TrainedGraphRetriever(
        graph_path=graph_path,
        model_path=model_path,
        ollama_model="llama3:8b",
        ollama_url="http://localhost:11434"
    )

    chat = InteractiveChatInterface(retriever)
    chat.run()


if __name__ == "__main__":
    main()