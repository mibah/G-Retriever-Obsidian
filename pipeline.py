#!/usr/bin/env python3
"""
Komplette Pipeline: Obsidian -> Graph -> Training Data -> G-Retriever
Führt alle Schritte automatisch aus.
"""

import argparse
from pathlib import Path
import sys


def run_pipeline(vault_path: str, output_dir: str, num_qa_pairs: int = 500,
                 ollama_model: str = "llama3:8b", skip_steps: list = None):
    """
    Führt komplette Pipeline aus:
    1. Obsidian -> Graph
    2. Graph -> Trainingsdaten (mit Ollama)
    3. Graph + Training -> PyG Dataset
    4. Inference Interface
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skip_steps = skip_steps or []

    graph_dir = output_dir / "graph_data"
    training_dir = output_dir / "training_data"
    processed_dir = output_dir / "processed_data"

    # Step 1: Obsidian -> Graph
    if "graph" not in skip_steps:
        print("\n" + "=" * 60)
        print("STEP 1: Converting Obsidian Vault to Graph")
        print("=" * 60)

        from obsidian_to_graph import ObsidianGraphBuilder

        builder = ObsidianGraphBuilder(vault_path)
        builder.build_graph()
        builder.save_graph(str(graph_dir))

        print(f"✓ Graph saved to {graph_dir}")
    else:
        print("\n[SKIPPING] Step 1: Graph conversion")

    graph_file = graph_dir / "graph.gpickle"
    if not graph_file.exists():
        print(f"ERROR: Graph file not found: {graph_file}")
        sys.exit(1)

    # Step 2: Generate Training Data
    if "training" not in skip_steps:
        print("\n" + "=" * 60)
        print("STEP 2: Generating Training Data with Ollama")
        print("=" * 60)

        from generate_training_data import TrainingDataGenerator

        generator = TrainingDataGenerator(
            str(graph_file),
            ollama_model=ollama_model
        )

        generator.generate_training_data(num_samples=num_qa_pairs)
        generator.save_training_data(str(training_dir))

        print(f"✓ Training data saved to {training_dir}")
    else:
        print("\n[SKIPPING] Step 2: Training data generation")

    train_file = training_dir / "train.json"
    val_file = training_dir / "val.json"

    if not train_file.exists() or not val_file.exists():
        print(f"ERROR: Training files not found in {training_dir}")
        sys.exit(1)

    # Step 3: Create PyG Datasets
    if "dataset" not in skip_steps:
        print("\n" + "=" * 60)
        print("STEP 3: Creating PyTorch Geometric Datasets")
        print("=" * 60)

        from pyg_dataset import create_datasets

        train_ds, val_ds = create_datasets(
            str(graph_file),
            str(train_file),
            str(val_file),
            root=str(processed_dir)
        )

        print(f"✓ Datasets created: {len(train_ds)} train, {len(val_ds)} val")
        print(f"✓ Processed data saved to {processed_dir}")
    else:
        print("\n[SKIPPING] Step 3: Dataset creation")

    # Step 4: Setup complete - ready for inference
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\nSetup Summary:")
    print(f"  Graph: {graph_file}")
    print(f"  Training: {training_dir}")
    print(f"  Datasets: {processed_dir}")

    print(f"\n🚀 Ready to chat! Run:")
    print(f"  python gretriever_inference.py {graph_file} --model {ollama_model}")

    return {
        "graph_file": str(graph_file),
        "train_dir": str(training_dir),
        "processed_dir": str(processed_dir)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Komplette Pipeline: Obsidian -> G-Retriever",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Komplette Pipeline
  python pipeline.py /path/to/vault --output ./g_retriever_data

  # Nur Graph erstellen
  python pipeline.py /path/to/vault -o ./data --skip training dataset

  # Mit weniger QA-Paaren (schneller)
  python pipeline.py /path/to/vault -o ./data -n 200

  # Custom Ollama Model
  python pipeline.py /path/to/vault -o ./data --model llama3:70b
        """
    )

    parser.add_argument("vault_path", help="Pfad zum Obsidian Vault")
    parser.add_argument("--output", "-o", default="./g_retriever_data",
                        help="Output-Verzeichnis (default: ./g_retriever_data)")
    parser.add_argument("--num-qa-pairs", "-n", type=int, default=500,
                        help="Anzahl QA-Paare (default: 500)")
    parser.add_argument("--model", "-m", default="llama3:8b",
                        help="Ollama Model (default: llama3:8b)")
    parser.add_argument("--skip", nargs="+",
                        choices=["graph", "training", "dataset"],
                        help="Schritte überspringen (z.B. --skip training)")

    args = parser.parse_args()

    # Validierung
    vault_path = Path(args.vault_path)
    if not vault_path.exists():
        print(f"ERROR: Vault path does not exist: {vault_path}")
        sys.exit(1)

    # Pipeline ausführen
    try:
        result = run_pipeline(
            str(vault_path),
            args.output,
            num_qa_pairs=args.num_qa_pairs,
            ollama_model=args.model,
            skip_steps=args.skip
        )

        print("\n✅ Success!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
