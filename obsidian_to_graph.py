"""
Obsidian Vault to Graph Converter
Wandelt Obsidian Notizen in eine Graph-Struktur für G-Retriever um.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, asdict
import networkx as nx


@dataclass
class Note:
    """Repräsentiert eine Obsidian-Notiz"""
    path: str
    title: str
    content: str
    links: List[str]  # Ausgehende Links
    tags: List[str]


class ObsidianGraphBuilder:
    """Baut einen Graph aus einem Obsidian Vault"""

    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        self.notes: Dict[str, Note] = {}
        self.graph = nx.DiGraph()

    def parse_note(self, file_path: Path) -> Note:
        """Parst eine einzelne Markdown-Datei"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Titel aus Dateinamen
        title = file_path.stem

        # Bilder entfernen (wie gewünscht)
        content = re.sub(r'!\[\[.*?\]\]', '', content)
        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)

        # Wiki-Links extrahieren [[Link]]
        wiki_links = re.findall(r'\[\[(.*?)\]\]', content)
        # Markdown-Links extrahieren [text](link)
        md_links = re.findall(r'\[.*?\]\((.*?\.md)\)', content)

        # Links bereinigen (Aliases entfernen [[Link|Alias]] -> Link)
        clean_links = []
        for link in wiki_links:
            clean_link = link.split('|')[0].strip()
            clean_links.append(clean_link)

        for link in md_links:
            clean_link = Path(link).stem
            clean_links.append(clean_link)

        # Tags extrahieren #tag
        tags = re.findall(r'#(\w+)', content)

        # Links für weitere Verarbeitung entfernen, aber Content behalten
        clean_content = re.sub(r'\[\[.*?\]\]', '', content)
        clean_content = re.sub(r'\[.*?\]\(.*?\)', '', clean_content)

        return Note(
            path=str(file_path.relative_to(self.vault_path)),
            title=title,
            content=clean_content.strip(),
            links=list(set(clean_links)),
            tags=list(set(tags))
        )

    def build_graph(self) -> nx.DiGraph:
        """Baut den kompletten Graphen aus allen Notizen"""
        print(f"Scanne Vault: {self.vault_path}")

        # Alle Markdown-Dateien finden
        md_files = list(self.vault_path.rglob("*.md"))
        print(f"Gefunden: {len(md_files)} Notizen")

        # Notizen parsen
        for file_path in md_files:
            try:
                note = self.parse_note(file_path)
                self.notes[note.title] = note

                # Knoten zum Graphen hinzufügen
                self.graph.add_node(
                    note.title,
                    content=note.content,
                    tags=note.tags,
                    path=note.path
                )
            except Exception as e:
                print(f"Fehler bei {file_path}: {e}")

        # Kanten hinzufügen (Links zwischen Notizen)
        for note in self.notes.values():
            for link in note.links:
                if link in self.notes:
                    self.graph.add_edge(note.title, link)

        print(f"Graph erstellt: {self.graph.number_of_nodes()} Knoten, "
              f"{self.graph.number_of_edges()} Kanten")

        # Minimal fix: Add a super node connecting to all notes to deal with unconnected nodes.
        super_node = "SUPER_NODE"
        self.graph.add_node(super_node, content="Artificial super node", tags=[], path="")
        for node in self.graph.nodes():
            if node != super_node:
                self.graph.add_edge(super_node, node)

        return self.graph

    def save_graph(self, output_path: str):
        """Speichert den Graphen in verschiedenen Formaten"""
        import pickle

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. NetworkX Graph (Pickle)
        with open(output_path / "graph.gpickle", 'wb') as f:
            pickle.dump(self.graph, f, pickle.HIGHEST_PROTOCOL)

        # 2. JSON für menschliche Lesbarkeit
        graph_data = {
            "nodes": [
                {
                    "id": node,
                    "content": data["content"][:10000],  # Erste 10000 Zeichen
                    "tags": data["tags"],
                    "path": data["path"]
                }
                for node, data in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v}
                for u, v in self.graph.edges()
            ]
        }

        with open(output_path / "graph.json", 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)

        # 3. Statistiken
        stats = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "avg_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
            "connected_components": nx.number_weakly_connected_components(self.graph),
            "tags": self._get_all_tags()
        }

        with open(output_path / "stats.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        print(f"Graph gespeichert in: {output_path}")
        print(f"Statistiken: {stats}")

    def _get_all_tags(self) -> Dict[str, int]:
        """Zählt alle Tags im Vault"""
        tag_counts = {}
        for _, data in self.graph.nodes(data=True):
            for tag in data.get("tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        return dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True))


def main():
    # todo Nur md files aus dem Vault laden, nicht sub folder, wie z.B. .trash

    vault_path = "/Users/mibahn/Documents/Obsidian_Vault_backup_11_dec_25"
    output_path = "./graph_output"

    builder = ObsidianGraphBuilder(vault_path)
    builder.build_graph()
    builder.save_graph(output_path)

if __name__ == "__main__":
    main()

