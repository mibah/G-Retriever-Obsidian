"""
PyTorch Geometric Dataset für G-Retriever
Konvertiert NetworkX Graph + QA-Paare in PyG Format.
"""

import json
import torch
import networkx as nx
from pathlib import Path
from typing import List, Dict, Optional
from torch_geometric.data import Data, InMemoryDataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class ObsidianGraphDataset(InMemoryDataset):
    """
    PyTorch Geometric Dataset für Obsidian Graph mit QA-Paaren
    """

    def __init__(self,
                 root: str,
                 graph_path: str,
                 qa_path: str,
                 split: str = "train",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 transform=None,
                 pre_transform=None):
        """
        Args:
            root: Root-Verzeichnis für processed data
            graph_path: Pfad zur graph.gpickle Datei
            qa_path: Pfad zu train.json oder val.json
            split: "train" oder "val"
            embedding_model: SentenceTransformer Model für Node Embeddings
        """
        self.graph_path = graph_path
        self.qa_path = qa_path
        self.split = split
        self.embedding_model_name = embedding_model

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'{self.split}_data.pt']

    def process(self):
        """Verarbeitet Graph und QA-Paare zu PyG Format"""
        import pickle

        print(f"Processing {self.split} dataset...")

        # 1. Lade Graph
        print("Loading graph...")
        with open(self.graph_path, 'rb') as f:
            graph = pickle.load(f)

        # 2. Lade QA-Paare
        print("Loading QA pairs...")
        with open(self.qa_path, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)

        # 3. Node Mapping (String -> Index)
        node_list = list(graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}

        # 4. Erstelle Node Features mit SentenceTransformer
        print("Creating node embeddings...")
        embedder = SentenceTransformer(self.embedding_model_name)

        node_texts = []
        for node in node_list:
            content = graph.nodes[node].get("content", "")
            title = node
            # Kombiniere Titel + Content (limitiert)
            text = f"{title}. {content[:500]}"
            node_texts.append(text)

        # Batch-Encoding für Effizienz
        node_embeddings = embedder.encode(
            node_texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_tensor=True
        )

        # 5. Erstelle Edge Index
        edge_list = list(graph.edges())
        edge_index = torch.tensor([
            [node_to_idx[u] for u, v in edge_list],
            [node_to_idx[v] for u, v in edge_list]
        ], dtype=torch.long)

        # Für undirected: beide Richtungen
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        # 6. Erstelle Data-Objekte für jedes QA-Paar
        print("Creating Data objects...")
        data_list = []

        for qa in tqdm(qa_pairs):
            question = qa["question"]
            answer = qa["answer"]

            # Question Embedding
            question_emb = embedder.encode(
                question,
                convert_to_tensor=True
            ).unsqueeze(0)  # [1, embed_dim]

            # Erstelle Data-Objekt
            data = Data(
                x=node_embeddings.clone(),  # Node features [num_nodes, embed_dim]
                edge_index=edge_index.clone(),  # Edges [2, num_edges]
                question=question,  # String
                answer=answer,  # String
                question_emb=question_emb,  # Tensor [1, embed_dim]
                num_nodes=len(node_list)
            )

            # Optional: Relevante Knoten markieren (für Training)
            if "node" in qa:
                # Single node
                relevant_nodes = [node_to_idx[qa["node"]]]
            elif "nodes" in qa:
                # Multiple nodes
                relevant_nodes = [node_to_idx[n] for n in qa["nodes"] if n in node_to_idx]
            else:
                relevant_nodes = []

            if relevant_nodes:
                data.relevant_nodes = torch.tensor(relevant_nodes, dtype=torch.long)

            data_list.append(data)

        # 7. Speichern
        self.save(data_list, self.processed_paths[0])
        print(f"Saved {len(data_list)} samples to {self.processed_paths[0]}")


class GraphTextDataset:
    """
    Alternative: Einfacheres Dataset ohne InMemoryDataset
    Gut für große Graphen die nicht ins RAM passen
    """

    def __init__(self, graph_path: str, qa_path: str,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        import pickle

        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)

        with open(qa_path, 'r', encoding='utf-8') as f:
            self.qa_pairs = json.load(f)

        self.node_list = list(self.graph.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.node_list)}

        self.embedder = SentenceTransformer(embedding_model)

        # Pre-compute node embeddings
        print("Pre-computing node embeddings...")
        node_texts = [
            f"{node}. {self.graph.nodes[node].get('content', '')[:500]}"
            for node in self.node_list
        ]
        self.node_embeddings = self.embedder.encode(
            node_texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_tensor=True
        )

        # Pre-compute edge index
        edge_list = list(self.graph.edges())
        edge_index = torch.tensor([
            [self.node_to_idx[u] for u, v in edge_list],
            [self.node_to_idx[v] for u, v in edge_list]
        ], dtype=torch.long)
        self.edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        qa = self.qa_pairs[idx]

        question_emb = self.embedder.encode(
            qa["question"],
            convert_to_tensor=True
        ).unsqueeze(0)

        data = Data(
            x=self.node_embeddings.clone(),
            edge_index=self.edge_index.clone(),
            question=qa["question"],
            answer=qa["answer"],
            question_emb=question_emb,
            num_nodes=len(self.node_list)
        )

        return data

    def get_node_text(self, node_idx: int) -> str:
        """Gibt Text für einen Knoten zurück"""
        node = self.node_list[node_idx]
        return self.graph.nodes[node].get("content", "")

    def get_subgraph_text(self, node_indices: List[int]) -> str:
        """Gibt kombinierten Text für mehrere Knoten zurück"""
        texts = []
        for idx in node_indices:
            node = self.node_list[idx]
            content = self.graph.nodes[node].get("content", "")
            texts.append(f"[{node}]: {content}")
        return "\n\n".join(texts)


def create_datasets(graph_path: str, train_path: str, val_path: str,
                    root: str = "./processed_data",
                    embedding_model: str = "all-MiniLM-L6-v2"):
    """
    Erstellt Train und Val Datasets

    Returns:
        train_dataset, val_dataset
    """
    train_dataset = ObsidianGraphDataset(
        root=root,
        graph_path=graph_path,
        qa_path=train_path,
        split="train",
        embedding_model=embedding_model
    )

    val_dataset = ObsidianGraphDataset(
        root=root,
        graph_path=graph_path,
        qa_path=val_path,
        split="val",
        embedding_model=embedding_model
    )

    return train_dataset, val_dataset


def main():
    """Direkte Nutzung"""

    graph_path = "./graph_output/graph.gpickle"
    train_path = "./training_data/train.json"
    val_path = "./training_data/val.json"
    output = "./processed_data"

    train_ds, val_ds = create_datasets(
        graph_path,
        train_path,
        val_path,
        root=output,
        embedding_model="all-MiniLM-L6-v2"
    )

    print(f"Train dataset: {len(train_ds)} samples")
    print(f"Val dataset: {len(val_ds)} samples")

if __name__ == "__main__":
    main()