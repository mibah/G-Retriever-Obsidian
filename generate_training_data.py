"""
Trainingsdaten Generator für G-Retriever
Erstellt Question-Answer Paare aus dem Obsidian-Graphen mit Ollama.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
import networkx as nx
import requests
from tqdm import tqdm


class TrainingDataGenerator:
    """Generiert QA-Paare für G-Retriever Training"""

    def __init__(self, graph_path: str, ollama_model: str = "llama3:8b",
                 ollama_url: str = "http://localhost:11434"):
        import pickle

        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)

        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.qa_pairs = []

    def generate_ollama_prompt(self, context: str, prompt_type: str) -> str:
        """Erstellt Prompts für verschiedene Fragetypen"""
        prompts = {
        "factual": f"""Basierend auf diesem Text, erstelle eine präzise Faktenfrage und die Antwort.
            Text: {context}
            
            Erstelle die Frage und Antwort im JSON-Format:
            {{"question": "...", "answer": "..."}}
            
            Nur JSON zurückgeben, keine Erklärung.""",

        "connection": f"""Basierend auf diesem Text, erstelle eine Frage über Zusammenhänge/Verbindungen und die Antwort.
            Text: {context}
            
            Erstelle die Frage und Antwort im JSON-Format:
            {{"question": "...", "answer": "..."}}
            
            Nur JSON zurückgeben, keine Erklärung.""",

        "summary": f"""Basierend auf diesem Text, erstelle eine Frage die nach einer Zusammenfassung fragt und die Antwort.
            Text: {context}
            
            Erstelle die Frage und Antwort im JSON-Format:
            {{"question": "...", "answer": "..."}}
            
            Nur JSON zurückgeben, keine Erklärung."""
        }
        return prompts[prompt_type]

    def call_ollama(self, prompt: str) -> str:
        """Ruft Ollama API auf"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Ollama Error: {e}")
            return None

    def extract_json_from_response(self, response: str) -> Dict:
        """Extrahiert JSON aus Ollama Response"""
        try:
            # Versuche direkt zu parsen
            return json.loads(response)
        except:
            # Falls Markdown-Blöcke vorhanden
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))

            # Falls ohne Blöcke aber mit {}
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))

        return None

    def generate_single_node_qa(self, node: str, question_type: str) -> Dict:
        """Generiert ein QA-Paar für einen einzelnen Knoten"""
        node_data = self.graph.nodes[node]
        content = node_data.get("content", "")

        # Zu kurze Notizen überspringen
        if len(content) < 100:
            return None

        # Kontext limitieren (Ollama Context Window)
        context = content[:2000]

        prompt = self.generate_ollama_prompt(context, question_type)
        response = self.call_ollama(prompt)

        if not response:
            return None

        qa_data = self.extract_json_from_response(response)
        if not qa_data or "question" not in qa_data or "answer" not in qa_data:
            return None

        return {
            "question": qa_data["question"],
            "answer": qa_data["answer"],
            "node": node,
            "type": question_type
        }

    def generate_multi_node_qa(self, nodes: List[str]) -> Dict:
        """Generiert QA-Paar über mehrere verbundene Knoten"""
        # Kombiniere Content von mehreren Knoten
        contexts = []
        for node in nodes[:3]:  # Max 3 Knoten
            content = self.graph.nodes[node].get("content", "")
            if content:
                contexts.append(f"[{node}]: {content[:500]}")

        combined_context = "\n\n".join(contexts)

        prompt = f"""Basierend auf diesen verbundenen Notizen, erstelle eine Frage die Wissen aus mehreren Notizen kombiniert und die Antwort.

{combined_context}

Erstelle die Frage und Antwort im JSON-Format:
{{"question": "...", "answer": "..."}}

Nur JSON zurückgeben, keine Erklärung."""

        response = self.call_ollama(prompt)
        if not response:
            return None

        qa_data = self.extract_json_from_response(response)
        if not qa_data or "question" not in qa_data or "answer" not in qa_data:
            return None

        return {
            "question": qa_data["question"],
            "answer": qa_data["answer"],
            "nodes": nodes,
            "type": "multi_node"
        }

    def generate_training_data(self, num_samples: int = 500,
                               multi_node_ratio: float = 0.3) -> List[Dict]:
        """Generiert vollständigen Trainingsdatensatz"""
        print(f"Generiere {num_samples} QA-Paare...")

        nodes = list(self.graph.nodes())
        num_multi = int(num_samples * multi_node_ratio)
        num_single = num_samples - num_multi

        question_types = ["factual", "connection", "summary"]

        # Single-Node QA-Paare
        print("Generiere Single-Node Fragen...")
        for _ in tqdm(range(num_single)):
            node = random.choice(nodes)
            q_type = random.choice(question_types)

            qa = self.generate_single_node_qa(node, q_type)
            if qa:
                self.qa_pairs.append(qa)

        # Multi-Node QA-Paare (über verbundene Knoten)
        print("Generiere Multi-Node Fragen...")
        for _ in tqdm(range(num_multi)):
            # Wähle Startknoten mit Nachbarn
            start_node = random.choice([n for n in nodes if self.graph.degree(n) > 0])
            neighbors = list(self.graph.neighbors(start_node))

            if len(neighbors) > 0:
                # Startknoten + 1-2 Nachbarn
                selected = [start_node] + random.sample(neighbors, min(2, len(neighbors)))
                qa = self.generate_multi_node_qa(selected)
                if qa:
                    self.qa_pairs.append(qa)

        print(f"Erfolgreich generiert: {len(self.qa_pairs)} QA-Paare")
        return self.qa_pairs

    def save_training_data(self, output_path: str):
        """Speichert Trainingsdaten"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Vollständiger Datensatz
        with open(output_path / "qa_pairs.json", 'w', encoding='utf-8') as f:
            json.dump(self.qa_pairs, f, indent=2, ensure_ascii=False)

        # Train/Val Split (80/20)
        random.shuffle(self.qa_pairs)
        split_idx = int(len(self.qa_pairs) * 0.8)

        train_data = self.qa_pairs[:split_idx]
        val_data = self.qa_pairs[split_idx:]

        with open(output_path / "train.json", 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)

        with open(output_path / "val.json", 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)

        # Statistiken
        stats = {
            "total": len(self.qa_pairs),
            "train": len(train_data),
            "val": len(val_data),
            "types": {
                "factual": sum(1 for qa in self.qa_pairs if qa.get("type") == "factual"),
                "connection": sum(1 for qa in self.qa_pairs if qa.get("type") == "connection"),
                "summary": sum(1 for qa in self.qa_pairs if qa.get("type") == "summary"),
                "multi_node": sum(1 for qa in self.qa_pairs if qa.get("type") == "multi_node")
            }
        }

        with open(output_path / "stats.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        print(f"Trainingsdaten gespeichert in: {output_path}")
        print(f"Statistiken: {stats}")


def main():
    """Beispiel-Nutzung ohne CLI-Argumente"""

    graph_path = "./graph_output/graph.gpickle"  # Pfad zur gpickle-Datei
    output_path = "./training_data"             # Speicherort für Trainingsdaten
    num_samples = 500                            # Anzahl QA-Paare
    model = "llama3:8b"                          # Ollama Model
    ollama_url = "http://localhost:11434"        # Ollama API URL

    generator = TrainingDataGenerator(
        graph_path,
        ollama_model=model,
        ollama_url=ollama_url
    )

    generator.generate_training_data(num_samples=num_samples)
    generator.save_training_data(output_path)


if __name__ == "__main__":
    main()