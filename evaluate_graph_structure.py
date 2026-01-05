"""
Umfassende Graph-Struktur Evaluation für GNN Training
Basiert auf Forschung zu GNN-Anforderungen und Best Practices
"""

import pickle
import json
import numpy as np
import networkx as nx
from pathlib import Path
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


class GraphEvaluator:
    """Bewertet Graph-Struktur für GNN Training"""

    def __init__(self, graph_path: str):
        print("Loading graph...")
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)

        self.num_nodes = len(self.graph.nodes())
        self.num_edges = len(self.graph.edges())

        # Make undirected for connectivity analysis
        self.graph_undirected = self.graph.to_undirected() if self.graph.is_directed() else self.graph

    def evaluate_all(self):
        """Führt alle Evaluationen durch"""
        print("\n" + "=" * 80)
        print("GRAPH NEURAL NETWORK STRUCTURE EVALUATION")
        print("=" * 80)

        results = {}

        # 1. Basic Statistics
        print("\n" + "─" * 80)
        print("1. BASIC GRAPH STATISTICS")
        print("─" * 80)
        results['basic'] = self.evaluate_basic_stats()

        # 2. Connectivity
        print("\n" + "─" * 80)
        print("2. GRAPH CONNECTIVITY (Critical for Message Passing)")
        print("─" * 80)
        results['connectivity'] = self.evaluate_connectivity()

        # 3. Degree Distribution
        print("\n" + "─" * 80)
        print("3. DEGREE DISTRIBUTION (Affects Information Flow)")
        print("─" * 80)
        results['degree'] = self.evaluate_degree_distribution()

        # 4. Density & Sparsity
        print("\n" + "─" * 80)
        print("4. GRAPH DENSITY (Message Passing Efficiency)")
        print("─" * 80)
        results['density'] = self.evaluate_density()

        # 5. Path Lengths
        print("\n" + "─" * 80)
        print("5. PATH LENGTHS (GNN Depth Requirements)")
        print("─" * 80)
        results['paths'] = self.evaluate_path_lengths()

        # 6. Clustering
        print("\n" + "─" * 80)
        print("6. CLUSTERING & COMMUNITY STRUCTURE")
        print("─" * 80)
        results['clustering'] = self.evaluate_clustering()

        # 7. Node Features Quality
        print("\n" + "─" * 80)
        print("7. NODE CONTENT QUALITY (Feature Signal)")
        print("─" * 80)
        results['features'] = self.evaluate_node_features()

        # 8. Over-smoothing Risk
        print("\n" + "─" * 80)
        print("8. OVER-SMOOTHING RISK ASSESSMENT")
        print("─" * 80)
        results['oversmoothing'] = self.evaluate_oversmoothing_risk()

        # 9. Over-squashing Risk
        print("\n" + "─" * 80)
        print("9. OVER-SQUASHING RISK (Bottlenecks)")
        print("─" * 80)
        results['oversquashing'] = self.evaluate_oversquashing_risk()

        # Final Summary
        print("\n" + "=" * 80)
        print("OVERALL ASSESSMENT & RECOMMENDATIONS")
        print("=" * 80)
        self.print_overall_assessment(results)

        return results

    def evaluate_basic_stats(self):
        """Grundlegende Statistiken"""
        stats = {
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'is_directed': self.graph.is_directed()
        }

        print(f"Nodes: {self.num_nodes}")
        print(f"Edges: {self.num_edges}")
        print(f"Directed: {stats['is_directed']}")

        return stats

    def evaluate_connectivity(self):
        """Connectivity - Critical für Message Passing"""
        # Connected components
        if self.graph_undirected.is_directed():
            n_components = nx.number_weakly_connected_components(self.graph)
        else:
            n_components = nx.number_connected_components(self.graph_undirected)

        components = list(nx.connected_components(self.graph_undirected))
        component_sizes = [len(c) for c in components]
        largest_component_size = max(component_sizes) if component_sizes else 0

        # Isolated nodes
        isolated = [n for n in self.graph.nodes() if self.graph.degree(n) == 0]
        n_isolated = len(isolated)

        results = {
            'n_components': n_components,
            'largest_component_size': largest_component_size,
            'largest_component_ratio': largest_component_size / self.num_nodes if self.num_nodes > 0 else 0,
            'n_isolated': n_isolated,
            'isolated_ratio': n_isolated / self.num_nodes if self.num_nodes > 0 else 0
        }

        print(f"Connected Components: {n_components}")
        print(f"Largest Component: {largest_component_size} nodes ({results['largest_component_ratio'] * 100:.1f}%)")
        print(f"Isolated Nodes: {n_isolated} ({results['isolated_ratio'] * 100:.1f}%)")

        # Assessment
        if n_components == 1:
            print("✓ GOOD: Graph is fully connected")
        elif results['largest_component_ratio'] > 0.9:
            print("⚠ WARNING: Graph is mostly connected but has small components")
        else:
            print("❌ PROBLEM: Graph is highly fragmented")
            print("   → GNNs work best on connected graphs")
            print("   → Consider adding more links between notes")

        if results['isolated_ratio'] > 0.1:
            print(f"❌ PROBLEM: {results['isolated_ratio'] * 100:.1f}% isolated nodes")
            print("   → Isolated nodes cannot participate in message passing")
            print("   → Will only rely on initial features")

        return results

    def evaluate_degree_distribution(self):
        """Degree Distribution - Affects information flow"""
        degrees = dict(self.graph.degree())
        degree_values = list(degrees.values())

        avg_degree = np.mean(degree_values)
        median_degree = np.median(degree_values)
        min_degree = min(degree_values)
        max_degree = max(degree_values)
        std_degree = np.std(degree_values)

        # Degree distribution
        degree_counts = Counter(degree_values)

        # Low-degree nodes (problematic)
        low_degree_threshold = 2
        n_low_degree = sum(1 for d in degree_values if d < low_degree_threshold)
        low_degree_ratio = n_low_degree / len(degree_values) if degree_values else 0

        results = {
            'avg_degree': avg_degree,
            'median_degree': median_degree,
            'min_degree': min_degree,
            'max_degree': max_degree,
            'std_degree': std_degree,
            'low_degree_ratio': low_degree_ratio
        }

        print(f"Average Degree: {avg_degree:.2f}")
        print(f"Median Degree: {median_degree:.1f}")
        print(f"Range: [{min_degree}, {max_degree}]")
        print(f"Std Dev: {std_degree:.2f}")
        print(f"Nodes with degree < 2: {n_low_degree} ({low_degree_ratio * 100:.1f}%)")

        # Assessment
        if avg_degree < 2:
            print("❌ CRITICAL: Average degree < 2")
            print("   → Graph is too sparse for effective message passing")
            print("   → Recommendation: Add more links (target: avg degree > 3)")
        elif avg_degree < 3:
            print("⚠ WARNING: Average degree < 3")
            print("   → Limited message passing capability")
            print("   → Consider adding more connections")
        else:
            print(f"✓ GOOD: Average degree = {avg_degree:.2f}")

        if low_degree_ratio > 0.3:
            print(f"⚠ WARNING: {low_degree_ratio * 100:.0f}% nodes have very few connections")
            print("   → These nodes will have poor representations")

        return results

    def evaluate_density(self):
        """Graph Density - Balance between sparse and dense"""
        density = nx.density(self.graph)
        max_possible_edges = self.num_nodes * (self.num_nodes - 1) / 2

        results = {
            'density': density,
            'max_possible_edges': max_possible_edges,
            'edge_ratio': self.num_edges / max_possible_edges if max_possible_edges > 0 else 0
        }

        print(f"Graph Density: {density:.6f}")
        print(f"Actual Edges: {self.num_edges}")
        print(f"Max Possible: {int(max_possible_edges)}")
        print(f"Edge Ratio: {results['edge_ratio'] * 100:.4f}%")

        # Assessment
        if density < 0.0001:
            print("❌ CRITICAL: Graph is extremely sparse")
            print("   → GNNs need sufficient connectivity")
            print("   → Typical knowledge graphs: 0.001-0.01 density")
        elif density < 0.001:
            print("⚠ WARNING: Graph is very sparse")
            print("   → May limit GNN effectiveness")
        else:
            print("✓ Density is acceptable for GNN training")

        return results

    def evaluate_path_lengths(self):
        """Path Lengths - Determines required GNN depth"""
        # Only analyze largest component
        largest_cc = max(nx.connected_components(self.graph_undirected), key=len)
        subgraph = self.graph_undirected.subgraph(largest_cc)

        # Sample for efficiency if too large
        if len(largest_cc) > 500:
            print("Sampling 500 nodes for path analysis...")
            sample_nodes = np.random.choice(list(largest_cc), 500, replace=False)
            subgraph = self.graph_undirected.subgraph(sample_nodes)

        try:
            # Average shortest path
            avg_path_length = nx.average_shortest_path_length(subgraph)
            diameter = nx.diameter(subgraph)

            # Distribution of path lengths
            path_lengths = []
            nodes = list(subgraph.nodes())[:100]  # Sample
            for i, source in enumerate(nodes):
                lengths = nx.single_source_shortest_path_length(subgraph, source)
                path_lengths.extend(lengths.values())

            results = {
                'avg_path_length': avg_path_length,
                'diameter': diameter,
                'path_lengths_sampled': path_lengths
            }

            print(f"Average Shortest Path: {avg_path_length:.2f}")
            print(f"Diameter (max path): {diameter}")

            # Recommended GNN depth
            recommended_depth = int(np.ceil(avg_path_length)) + 1
            print(f"\n→ Recommended GNN Depth: {recommended_depth} layers")
            print(f"   (Based on avg path length + 1)")

            # Assessment
            if avg_path_length > 10:
                print("⚠ WARNING: Very long paths in graph")
                print("   → May need deep GNN (risk of over-smoothing)")
                print("   → Consider adding shortcuts between distant nodes")
            elif avg_path_length < 2:
                print("⚠ Note: Very short paths")
                print("   → Graph may be too densely connected")
            else:
                print("✓ Path lengths suitable for GNN")

        except nx.NetworkXError:
            results = {'error': 'Graph not connected enough for path analysis'}
            print("⚠ Could not compute paths (graph too disconnected)")

        return results

    def evaluate_clustering(self):
        """Clustering & Community Structure"""
        # Clustering coefficient
        avg_clustering = nx.average_clustering(self.graph_undirected)

        # Transitivity (global clustering)
        transitivity = nx.transitivity(self.graph_undirected)

        results = {
            'avg_clustering': avg_clustering,
            'transitivity': transitivity
        }

        print(f"Average Clustering Coefficient: {avg_clustering:.4f}")
        print(f"Transitivity: {transitivity:.4f}")

        # Assessment
        if avg_clustering < 0.01:
            print("⚠ WARNING: Very low clustering")
            print("   → Graph lacks local structure")
            print("   → GNNs may not capture meaningful patterns")
        elif avg_clustering > 0.3:
            print("✓ GOOD: Strong clustering (local structure present)")
        else:
            print("✓ Moderate clustering present")

        return results

    def evaluate_node_features(self):
        """Node Feature Quality"""
        # Content length distribution
        content_lengths = []
        empty_nodes = 0

        for node in self.graph.nodes():
            content = self.graph.nodes[node].get('content', '')
            length = len(content)
            content_lengths.append(length)
            if length == 0:
                empty_nodes += 1

        avg_length = np.mean(content_lengths)
        median_length = np.median(content_lengths)
        min_length = min(content_lengths)
        max_length = max(content_lengths)

        # Short content nodes
        short_threshold = 100
        n_short = sum(1 for l in content_lengths if l < short_threshold)
        short_ratio = n_short / len(content_lengths) if content_lengths else 0

        results = {
            'avg_content_length': avg_length,
            'median_content_length': median_length,
            'min_content_length': min_length,
            'max_content_length': max_length,
            'empty_nodes': empty_nodes,
            'short_content_ratio': short_ratio
        }

        print(f"Average Content Length: {avg_length:.0f} chars")
        print(f"Median: {median_length:.0f} chars")
        print(f"Range: [{min_length}, {max_length}]")
        print(f"Empty Nodes: {empty_nodes}")
        print(f"Nodes < 100 chars: {n_short} ({short_ratio * 100:.1f}%)")

        # Assessment
        if empty_nodes > self.num_nodes * 0.1:
            print(f"⚠ WARNING: {empty_nodes} nodes have no content")
            print("   → These nodes lack feature signal")

        if short_ratio > 0.5:
            print(f"⚠ WARNING: {short_ratio * 100:.0f}% nodes have little content")
            print("   → Weak feature signal for embeddings")
        elif avg_length > 200:
            print("✓ GOOD: Nodes have substantial content")

        return results

    def evaluate_oversmoothing_risk(self):
        """Over-smoothing Risk - nodes become indistinguishable"""
        # Risk factors:
        # 1. High average degree
        # 2. High density
        # 3. Low clustering

        degrees = [d for n, d in self.graph.degree()]
        avg_degree = np.mean(degrees)
        density = nx.density(self.graph)
        clustering = nx.average_clustering(self.graph_undirected)

        # Risk score (higher = more risk)
        risk_score = 0

        if avg_degree > 10:
            risk_score += 2
        elif avg_degree > 5:
            risk_score += 1

        if density > 0.01:
            risk_score += 2
        elif density > 0.005:
            risk_score += 1

        if clustering < 0.05:
            risk_score += 1

        results = {
            'risk_score': risk_score,
            'factors': {
                'avg_degree': avg_degree,
                'density': density,
                'clustering': clustering
            }
        }

        print(f"Over-smoothing Risk Score: {risk_score}/5")

        if risk_score >= 4:
            print("❌ HIGH RISK of over-smoothing")
            print("   → Use shallow GNN (2-3 layers max)")
            print("   → Consider residual connections")
            print("   → Add skip connections or normalization")
        elif risk_score >= 2:
            print("⚠ MODERATE RISK of over-smoothing")
            print("   → Limit GNN depth to 3-4 layers")
            print("   → Monitor layer-wise similarity")
        else:
            print("✓ LOW RISK: Graph structure is diverse enough")

        return results

    def evaluate_oversquashing_risk(self):
        """Over-squashing Risk - information bottlenecks"""
        # Find articulation points (bottlenecks)
        articulation_points = list(nx.articulation_points(self.graph_undirected))
        n_bottlenecks = len(articulation_points)

        # Find bridges (edges whose removal disconnects graph)
        bridges = list(nx.bridges(self.graph_undirected))
        n_bridges = len(bridges)

        # Degree variance (high variance = bottlenecks)
        degrees = [d for n, d in self.graph.degree()]
        degree_variance = np.var(degrees)

        results = {
            'n_bottlenecks': n_bottlenecks,
            'n_bridges': n_bridges,
            'degree_variance': degree_variance,
            'bottleneck_ratio': n_bottlenecks / self.num_nodes if self.num_nodes > 0 else 0
        }

        print(f"Articulation Points (bottlenecks): {n_bottlenecks}")
        print(f"Bridges (critical edges): {n_bridges}")
        print(f"Degree Variance: {degree_variance:.2f}")

        if n_bottlenecks > self.num_nodes * 0.1:
            print("❌ HIGH RISK of over-squashing")
            print(f"   → {n_bottlenecks} critical bottleneck nodes")
            print("   → Information flow is restricted")
            print("   → Recommendation: Add redundant paths")
        elif n_bottlenecks > 0:
            print("⚠ MODERATE RISK: Some bottlenecks present")
            print("   → Consider graph rewiring techniques")
        else:
            print("✓ LOW RISK: No obvious bottlenecks")

        return results

    def print_overall_assessment(self, results):
        """Final assessment and recommendations"""
        issues = []
        warnings = []

        # Check critical issues
        if results['connectivity']['n_components'] > 1:
            issues.append("Graph is fragmented (multiple components)")

        if results['degree']['avg_degree'] < 2:
            issues.append("Average degree too low (< 2)")

        if results['connectivity']['isolated_ratio'] > 0.1:
            issues.append(f"{results['connectivity']['isolated_ratio'] * 100:.0f}% isolated nodes")

        if results['density']['density'] < 0.0001:
            issues.append("Graph is extremely sparse")

        # Check warnings
        if results['degree']['low_degree_ratio'] > 0.3:
            warnings.append("Many nodes have low degree")

        if results['clustering']['avg_clustering'] < 0.01:
            warnings.append("Very low clustering coefficient")

        if results['oversmoothing']['risk_score'] >= 3:
            warnings.append("High over-smoothing risk")

        # Print assessment
        print()
        if not issues and not warnings:
            print("✓✓✓ EXCELLENT: Graph structure is well-suited for GNN training!")
            print()
            print("Your graph has:")
            print("  • Good connectivity")
            print("  • Sufficient density")
            print("  • Appropriate degree distribution")
            print("  • Low risk of over-smoothing/over-squashing")
        else:
            if issues:
                print("❌ CRITICAL ISSUES FOUND:")
                for i, issue in enumerate(issues, 1):
                    print(f"  {i}. {issue}")
                print()

            if warnings:
                print("⚠ WARNINGS:")
                for i, warning in enumerate(warnings, 1):
                    print(f"  {i}. {warning}")
                print()

            print("RECOMMENDATIONS:")
            print()

            if results['degree']['avg_degree'] < 2:
                print("→ PRIORITY 1: Increase graph connectivity")
                print("  • Add more links between related notes")
                print("  • Use bidirectional links")
                print("  • Consider semantic similarity-based edges")
                print("  • Target: Average degree > 3")
                print()

            if results['connectivity']['isolated_ratio'] > 0.1:
                print("→ PRIORITY 2: Connect isolated nodes")
                print("  • Find semantic connections for isolated notes")
                print("  • Consider hub nodes to bridge components")
                print()

            if results['oversmoothing']['risk_score'] >= 3:
                print("→ Use shallow GNN architecture")
                print("  • 2-3 layers maximum")
                print("  • Add residual connections")
                print("  • Use layer normalization")
                print()

            if 'avg_path_length' in results['paths']:
                depth = int(np.ceil(results['paths']['avg_path_length'])) + 1
                print(f"→ Recommended GNN depth: {depth} layers")
                print()


def main():
    """Run evaluation"""
    graph_path = "./graph_output/graph.gpickle"

    if not Path(graph_path).exists():
        print(f"Error: Graph file not found at {graph_path}")
        return

    evaluator = GraphEvaluator(graph_path)
    results = evaluator.evaluate_all()

    # Save results
    output_path = Path("./graph_evaluation_results.json")

    # Convert numpy types to native Python for JSON
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    results_serializable = convert_numpy(results)

    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()