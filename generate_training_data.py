"""
Trainingsdaten Generator für G-Retriever
Erstellt Question-Answer Paare aus dem Obsidian-Graphen mit Ollama.
FIXED: Speichert jetzt auch relevant_nodes für Training!
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

        # Node mapping für Training
        self.node_list = list(self.graph.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.node_list)}

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
        import re

        # Entferne führende/trailing whitespace
        response = response.strip()

        # Versuche 1: Direkt parsen
        try:
            return json.loads(response)
        except:
            pass

        # Versuche 2: Markdown-Blöcke
        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
        except:
            pass

        # Versuche 3: Finde ersten vollständigen JSON-Block
        try:
            # Finde alle { ... } Paare
            stack = []
            start = -1
            for i, char in enumerate(response):
                if char == '{':
                    if not stack:
                        start = i
                    stack.append(char)
                elif char == '}':
                    if stack:
                        stack.pop()
                        if not stack and start != -1:
                            # Vollständiger Block gefunden
                            json_str = response[start:i+1]
                            try:
                                return json.loads(json_str)
                            except:
                                # Versuche nächsten Block
                                continue
        except:
            pass

        # Versuche 4: Regex Fallback
        try:
            json_match = re.search(r'\{[^{}]*"question"[^{}]*"answer"[^{}]*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except:
            pass

        return None

    def generate_single_node_qa(self, node: str, question_type: str) -> Dict:
        """Generiert ein QA-Paar für einen einzelnen Knoten"""
        try:
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

            # WICHTIG: Speichere Node-Namen UND Indices!
            return {
                "question": qa_data["question"],
                "answer": qa_data["answer"],
                "node": node,
                "relevant_nodes": [node],  # Als Liste für Konsistenz
                "relevant_node_indices": [self.node_to_idx[node]],  # Indices für Training
                "type": question_type
            }
        except Exception as e:
            # Logge Fehler aber breche nicht ab
            print(f"\nError generating QA for node '{node}': {e}")
            return None

    def generate_multi_node_qa(self, nodes: List[str]) -> Dict:
        """Generiert QA-Paar über mehrere verbundene Knoten"""
        try:
            # Kombiniere Content von mehreren Knoten
            contexts = []
            for node in nodes[:3]:  # Max 3 Knoten
                content = self.graph.nodes[node].get("content", "")
                if content:
                    contexts.append(f"[{node}]: {content[:500]}")

            if not contexts:
                return None

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

            # WICHTIG: Speichere alle relevanten Nodes!
            return {
                "question": qa_data["question"],
                "answer": qa_data["answer"],
                "nodes": nodes,
                "relevant_nodes": nodes,
                "relevant_node_indices": [self.node_to_idx[n] for n in nodes],
                "type": "multi_node"
            }
        except Exception as e:
            print(f"\nError generating multi-node QA: {e}")
            return None

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
        attempts = 0
        max_attempts = num_single * 3  # Max 3x versuchen

        pbar = tqdm(total=num_single)
        while len([qa for qa in self.qa_pairs if qa.get("type") != "multi_node"]) < num_single and attempts < max_attempts:
            node = random.choice(nodes)
            q_type = random.choice(question_types)

            qa = self.generate_single_node_qa(node, q_type)
            if qa:
                self.qa_pairs.append(qa)
                pbar.update(1)
            attempts += 1
        pbar.close()

        # Multi-Node QA-Paare (über verbundene Knoten)
        print("Generiere Multi-Node Fragen...")
        attempts = 0
        max_attempts = num_multi * 3

        pbar = tqdm(total=num_multi)
        while len([qa for qa in self.qa_pairs if qa.get("type") == "multi_node"]) < num_multi and attempts < max_attempts:
            # Wähle Startknoten mit Nachbarn
            nodes_with_neighbors = [n for n in nodes if self.graph.degree(n) > 0]
            if not nodes_with_neighbors:
                break

            start_node = random.choice(nodes_with_neighbors)
            neighbors = list(self.graph.neighbors(start_node))

            if len(neighbors) > 0:
                # Startknoten + 1-2 Nachbarn
                selected = [start_node] + random.sample(neighbors, min(2, len(neighbors)))
                qa = self.generate_multi_node_qa(selected)
                if qa:
                    self.qa_pairs.append(qa)
                    pbar.update(1)
            attempts += 1
        pbar.close()

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
            },
            "avg_relevant_nodes": sum(len(qa.get("relevant_nodes", [])) for qa in self.qa_pairs) / len(self.qa_pairs) if self.qa_pairs else 0
        }

        with open(output_path / "stats.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        print(f"Trainingsdaten gespeichert in: {output_path}")
        print(f"Statistiken: {stats}")


def main():
    """Beispiel-Nutzung ohne CLI-Argumente"""

    graph_path = "./graph_output/graph.gpickle"  # Pfad zur gpickle-Datei
    output_path = "./training_data"             # Speicherort für Trainingsdaten
    num_samples = 2000                            # Anzahl QA-Paare
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