"""
G-Retriever Inference mit TRAINIERTEM Model

Das Model konkateniert
"""

import torch
from gretriever_inference import GraphRetriever, InteractiveChatInterface
from train_gretriever import GRetrieverModel


class TrainedGraphRetriever(GraphRetriever):
    """GraphRetriever mit trainiertem GNN"""

    def __init__(self, graph_path: str, model_path: str, **kwargs):
        super().__init__(graph_path, **kwargs)

        # Lade trainiertes Model
        self._log(f"Loading trained model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)

        embed_dim = self.node_embeddings.shape[1]
        self.trained_model = GRetrieverModel(
            node_embed_dim=embed_dim,
            hidden_dim=128,
            num_layers=2,
            num_heads=2
        ).to(self.device)

        self.trained_model.load_state_dict(checkpoint['model_state_dict'])
        self.trained_model.eval()

        self._log("✓ Trained model loaded!")

    def retrieve_relevant_nodes(self, query: str, k: int = 10):
        """Retrieval mit trainiertem Model"""
        query_emb = self.embedder.encode(
            query,
            convert_to_tensor=True,
            device=self.device
        )

        # DEBUG: Print query info
        if self.verbose:
            self._log(f"DEBUG: Query: '{query}'")
            self._log(f"DEBUG: Query embedding shape: {query_emb.shape}")

        # Nutze trainiertes Model für scoring
        with torch.no_grad():
            # Move everything to same device
            x = self.node_embeddings.to(self.device)
            edge_index = self.edge_index.to(self.device)
            q_emb = query_emb.unsqueeze(0).to(self.device)

            # Call model
            scores = self.trained_model(x, edge_index, q_emb, batch=None)

            # DEBUG: Print model output
            if self.verbose:
                self._log(f"DEBUG: Model output shape: {scores.shape}")
                self._log(f"DEBUG: Model output dim: {scores.dim()}")
                self._log(f"DEBUG: Expected num_nodes: {len(self.node_list)}")

                if scores.dim() == 1:
                    self._log(f"DEBUG: Output is 1D with {scores.shape[0]} elements")
                    if scores.shape[0] != len(self.node_list):
                        self._log(f"ERROR: Shape mismatch! {scores.shape[0]} != {len(self.node_list)}")
                    else:
                        self._log(f"DEBUG: Shape is correct!")
                        self._log(f"DEBUG: Score range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
                        self._log(f"DEBUG: Score mean: {scores.mean().item():.4f}")
                else:
                    self._log(f"ERROR: Output is not 1D! Shape: {scores.shape}")

            # Check if shape is correct
            if scores.dim() != 1 or scores.shape[0] != len(self.node_list):
                if self.verbose:
                    self._log("Warning: Model output dimension mismatch, falling back to cosine similarity")

                # Fallback to cosine similarity
                import torch.nn.functional as F
                scores = F.cosine_similarity(
                    query_emb.unsqueeze(0),
                    self.node_embeddings,
                    dim=1
                )

        # Top-k
        top_k_indices = torch.topk(scores, k=min(k, len(scores))).indices

        if self.verbose:
            top_scores = scores[top_k_indices].cpu()
            self._log(f"DEBUG: Top-5 indices: {top_k_indices[:5].cpu().tolist()}")
            self._log(f"DEBUG: Top-5 scores: {top_scores[:5].tolist()}")

        return top_k_indices.cpu().tolist()


def main():
    """Chat mit trainiertem Model"""

    graph_path = "./graph_output/graph.gpickle"

    # to load a preprocessed graph
    #graph_path = "./graph_output/graph_enhanced.gpickle"

    model_path = "./models/best_model.pt"  # Oder final_model.pt

    retriever = TrainedGraphRetriever(
        graph_path=graph_path,
        model_path=model_path,
        ollama_model="llama3:8b",
        ollama_url="http://localhost:11434",
        verbose=True  # Terminal output wie bei normalem Retriever
    )

    chat = InteractiveChatInterface(retriever)
    chat.run()


if __name__ == "__main__":
    main()