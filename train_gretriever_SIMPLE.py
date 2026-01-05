"""
G-Retriever Training Script
Trainiert das GNN auf deinen QA-Paaren

This model works fine. It uses a simple GAT as a model but is subsituted for a GAT with Cross-Attention.
"""

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from tqdm import tqdm
from pathlib import Path
import json


class GRetrieverModel(torch.nn.Module):
    """
    G-Retriever Model: GNN + Answer Generation
    """

    def __init__(self,
                 node_embed_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 num_heads: int = 2):
        super().__init__()

        # GNN Encoder (GAT)
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(node_embed_dim, hidden_dim, heads=num_heads, concat=True))

        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True)
            )

        # Projection layers
        self.node_proj = torch.nn.Linear(hidden_dim * num_heads, hidden_dim)
        self.question_proj = torch.nn.Linear(node_embed_dim, hidden_dim)

        # Relevance scorer
        self.relevance_scorer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, question_emb, batch=None):
        """
        Args:
            x: Node features [num_nodes, embed_dim]
            edge_index: Graph edges [2, num_edges]
            question_emb: Question embedding [batch_size, embed_dim]
            batch: Batch assignment for nodes (optional)

        Returns:
            node_scores: Relevance scores für jeden Knoten [num_nodes]
        """
        # GNN encoding
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)

        # Project node embeddings
        node_emb = self.node_proj(h)  # [num_nodes, hidden_dim]

        # Project question
        q_emb = self.question_proj(question_emb.squeeze(0))  # [hidden_dim]

        # Expand question to match all nodes
        q_expanded = q_emb.unsqueeze(0).expand(node_emb.size(0), -1)  # [num_nodes, hidden_dim]

        # Concatenate and score
        combined = torch.cat([node_emb, q_expanded], dim=1)  # [num_nodes, hidden_dim*2]
        scores = self.relevance_scorer(combined).squeeze(-1)  # [num_nodes]

        return scores


class GRetrieverTrainer:
    """Training Loop für G-Retriever"""

    def __init__(self,
                 model: GRetrieverModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 lr: float = 0.001):

        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        #self.criterion = torch.nn.BCEWithLogitsLoss()
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50.0]))

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for data in tqdm(self.train_loader, desc="Training"):
            data = data.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            scores = self.model(data.x, data.edge_index, data.question_emb)

            # Create target mit Hard Negatives
            target = torch.zeros(data.num_nodes, device=self.device)

            if hasattr(data, 'relevant_nodes'):
                # Positive samples
                target[data.relevant_nodes] = 1.0

                # Hard negative mining: Sample top-k falsch klassifizierte
                with torch.no_grad():
                    # Finde Nodes mit hohen Scores aber nicht relevant
                    mask = torch.ones(data.num_nodes, dtype=torch.bool)
                    mask[data.relevant_nodes] = False

                    hard_negatives = scores[mask].topk(min(10, mask.sum())).indices
                    # Gewichte diese stärker in Loss

            # Weighted BCE Loss
            pos_weight = torch.tensor([50.0], device=self.device)
            loss = self.criterion(scores, target)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        """Validierung"""
        self.model.eval()
        total_loss = 0

        for data in tqdm(self.val_loader, desc="Validation"):
            data = data.to(self.device)

            scores = self.model(data.x, data.edge_index, data.question_emb)

            target = torch.zeros(data.num_nodes, device=self.device)
            if hasattr(data, 'relevant_nodes'):
                target[data.relevant_nodes] = 1.0

            loss = self.criterion(scores, target)
            total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss

    def train(self, num_epochs: int = 10, save_dir: str = "./models"):
        """Komplettes Training mit Early Stopping"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float('inf')
        patience = 5  # Stoppe nach 5 Epochen ohne Verbesserung
        patience_counter = 0

        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Early stopping patience: {patience}")

        for epoch in range(num_epochs):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'=' * 60}")

            # Train
            train_loss = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}")

            # Validate
            val_loss = self.validate()
            print(f"Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, save_dir / "best_model.pt")
                print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"No improvement ({patience_counter}/{patience})")

                if patience_counter >= patience:
                    print(f"\n⚠️  Early stopping triggered after {epoch + 1} epochs")
                    print(f"Best val loss was: {best_val_loss:.4f}")
                    break

        # Save final model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, save_dir / "final_model.pt")

        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        with open(save_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)

        print(f"\n{'=' * 60}")
        print(f"Training complete!")
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"Models saved to: {save_dir}")
        print(f"{'=' * 60}\n")

def main():
    """Training ausführen"""
    from pyg_dataset import ObsidianGraphDataset

    # Pfade
    processed_dir = "./processed_data"
    graph_path = "./graph_output/graph.gpickle"
    train_path = "./training_data/train.json"
    val_path = "./training_data/val.json"

    print("Loading datasets...")
    train_dataset = ObsidianGraphDataset(
        root=processed_dir,
        graph_path=graph_path,
        qa_path=train_path,
        split="train"
    )

    val_dataset = ObsidianGraphDataset(
        root=processed_dir,
        graph_path=graph_path,
        qa_path=val_path,
        split="val"
    )

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Model
    sample_data = train_dataset[0]
    node_embed_dim = sample_data.x.shape[1]

    print(f"\nInitializing model...")
    print(f"Node embedding dim: {node_embed_dim}")

    model = GRetrieverModel(
        node_embed_dim=node_embed_dim,
        hidden_dim=128,
        num_layers=2,
        num_heads=2
    )

    # Trainer
    trainer = GRetrieverTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=0.001
    )

    # Train!
    trainer.train(num_epochs=20, save_dir="./models")


if __name__ == "__main__":
    main()