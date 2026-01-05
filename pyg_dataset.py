"""
PyTorch Geometric Dataset für G-Retriever
Konvertiert NetworkX Graph + QA-Paare in PyG Format.
FIXED: Nutzt jetzt relevant_node_indices aus den Trainingsdaten!
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
        if len(edge_list) > 0:
            edge_index = torch.tensor([
                [node_to_idx[u] for u, v in edge_list],
                [node_to_idx[v] for u, v in edge_list]
            ], dtype=torch.long)

            # Für undirected: beide Richtungen
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        else:
            edge_index = torch.tensor([[],[]], dtype=torch.long)

        # 6. Erstelle Data-Objekte für jedes QA-Paar
        print("Creating Data objects...")
        data_list = []
        skipped = 0

        for qa in tqdm(qa_pairs):
            question = qa["question"]
            answer = qa["answer"]

            # Question Embedding
            question_emb = embedder.encode(
                question,
                convert_to_tensor=True
            ).unsqueeze(0)  # [1, embed_dim]

            # WICHTIG: Hole relevant_node_indices aus QA-Daten
            relevant_nodes = None
            if "relevant_node_indices" in qa and qa["relevant_node_indices"]:
                # Direkt die Indices aus den Trainingsdaten
                relevant_nodes = torch.tensor(qa["relevant_node_indices"], dtype=torch.long)
            elif "relevant_nodes" in qa and qa["relevant_nodes"]:
                # Fallback: Konvertiere Node-Namen zu Indices
                relevant_nodes = torch.tensor(
                    [node_to_idx[n] for n in qa["relevant_nodes"] if n in node_to_idx],
                    dtype=torch.long
                )
            elif "node" in qa:
                # Alte Daten: Single node
                if qa["node"] in node_to_idx:
                    relevant_nodes = torch.tensor([node_to_idx[qa["node"]]], dtype=torch.long)
            elif "nodes" in qa:
                # Alte Daten: Multiple nodes
                relevant_nodes = torch.tensor(
                    [node_to_idx[n] for n in qa["nodes"] if n in node_to_idx],
                    dtype=torch.long
                )

            if relevant_nodes is None or len(relevant_nodes) == 0:
                print(f"Warning: No relevant nodes for question: {question[:50]}...")
                skipped += 1
                continue

            # Erstelle Data-Objekt
            data = Data(
                x=node_embeddings.clone(),  # Node features [num_nodes, embed_dim]
                edge_index=edge_index.clone(),  # Edges [2, num_edges]
                question=question,  # String
                answer=answer,  # String
                question_emb=question_emb,  # Tensor [1, embed_dim]
                relevant_nodes=relevant_nodes,  # Tensor [num_relevant_nodes]
                num_nodes=len(node_list)
            )

            data_list.append(data)

        # 7. Speichern
        self.save(data_list, self.processed_paths[0])
        print(f"Saved {len(data_list)} samples to {self.processed_paths[0]}")
        if skipped > 0:
            print(f"Skipped {skipped} samples without relevant nodes")


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

    # to load a preprocessed graph
    #graph_path = "./graph_output/graph_enhanced.gpickle"

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