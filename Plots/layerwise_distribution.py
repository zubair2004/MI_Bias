#!/usr/bin/env python3
"""
Layer-wise Distribution Visualization
Visualizes the distribution of important edges across model layers.

Usage:
    python Plots/layerwise_distribution.py <model_name> <results_dir> [--output_dir DIR] [--top_k K] [--threshold THRESHOLD]

Arguments:
    model_name: Model name (e.g., 'gpt2', 'gpt2-large')
    results_dir: Directory containing circuit finder results
    --output_dir: Output directory for plots (default: current directory)
    --top_k: Number of top edges to consider (default: 100)
    --threshold: Minimum percentage of edges in a layer to display (default: 0.02 = 2%)
"""

import sys
import os
import argparse
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_root not in sys.path:
    sys.path.append(project_root)


def parse_edge_name(edge_name):
    """Parse edge name to extract layer information."""
    # Edge format: node1->node2 or node1->node2<qkv>
    # Node format: a{layer}.h{head} or m{layer} or input or logits
    
    # Extract layer from attention nodes (a{layer}.h{head})
    attn_match = re.search(r'a(\d+)\.h\d+', edge_name)
    if attn_match:
        return int(attn_match.group(1))
    
    # Extract layer from MLP nodes (m{layer})
    mlp_match = re.search(r'm(\d+)', edge_name)
    if mlp_match:
        return int(mlp_match.group(1))
    
    # Input is layer 0
    if 'input' in edge_name.lower():
        return 0
    
    # Logits is last layer (will be handled separately)
    if 'logits' in edge_name.lower():
        return -1  # Special marker for logits
    
    return None


def load_top_edges(edges_file, top_k=100):
    """Load top K edges from a results file."""
    edges = []
    try:
        with open(edges_file, 'r') as f:
            lines = f.readlines()
            for line in lines[2:]:  # Skip header lines
                if line.strip() and ':' in line:
                    try:
                        # Parse edge ID from format: "1. edge_name: score=..."
                        parts = line.split(':')
                        if len(parts) >= 1:
                            edge_part = parts[0].strip()
                            # Extract edge name after the number and dot
                            if '.' in edge_part:
                                edge_id = edge_part.split('.', 1)[1].strip()
                                if edge_id:
                                    edges.append(edge_id)
                                    if len(edges) >= top_k:
                                        break
                    except (IndexError, ValueError) as e:
                        # Skip malformed lines
                        continue
    except Exception as e:
        print(f"Warning: Could not load edges from {edges_file}: {e}")
    return edges[:top_k]


def compute_layer_distribution(edges, n_layers):
    """Compute distribution of edges across layers."""
    layer_counts = defaultdict(int)
    total = 0
    
    for edge in edges:
        layer = parse_edge_name(edge)
        if layer is not None:
            if layer == -1:  # Logits
                layer = n_layers - 1
            layer_counts[layer] += 1
            total += 1
    
    # Convert to percentages
    layer_percentages = {}
    for layer in range(n_layers):
        count = layer_counts[layer]
        percentage = (count / total * 100) if total > 0 else 0
        layer_percentages[layer] = percentage
    
    return layer_percentages, layer_counts


def plot_layer_distribution(layer_percentages, config_name, output_file, threshold=2.0):
    """Plot layer-wise distribution of edges."""
    layers = sorted([l for l, p in layer_percentages.items() if p >= threshold])
    percentages = [layer_percentages[l] for l in layers]
    
    if not layers:
        print(f"Warning: No layers above {threshold}% threshold for {config_name}")
        return
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(layers, percentages, color='steelblue', alpha=0.7, edgecolor='black')
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Layer', fontsize=12, fontweight='bold')
    plt.ylabel('Percentage of Important Edges', fontsize=12, fontweight='bold')
    plt.title(f'Layer-wise Distribution of Important Edges\n{config_name}', 
              fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(layers, rotation=45)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Layer distribution plot saved to: {output_file}")


def get_model_n_layers(model_name):
    """Get number of layers for a model."""
    # Common model configurations
    model_configs = {
        'gpt2': 12,
        'gpt2-medium': 24,
        'gpt2-large': 36,
        'gpt2-xl': 48,
        'llama2-7B': 32,
        'llama2-13B': 40,
        'qwen2-0.5b': 24,
        'gemma-2-2b': 18,
    }
    
    for key, n_layers in model_configs.items():
        if key in model_name.lower():
            return n_layers
    
    # Default fallback
    print(f"Warning: Unknown model {model_name}, using default 12 layers")
    return 12


def main():
    parser = argparse.ArgumentParser(
        description="Layer-wise Distribution Visualization"
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Model name (e.g., 'gpt2', 'gpt2-large')"
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Directory containing circuit finder results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for plots (default: current directory)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Number of top edges to consider (default: 100)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Minimum percentage of edges in a layer to display (default: 2.0%%)"
    )

    args = parser.parse_args()

    # Get model configuration
    n_layers = get_model_n_layers(args.model_name)
    
    # Find all result files
    pattern = os.path.join(args.results_dir, f"{args.model_name}_*_*_*_top_edges.txt")
    files = glob.glob(pattern)
    
    if len(files) == 0:
        print(f"Error: No result files found in {args.results_dir}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Processing {len(files)} result files...")
    
    for edges_file in files:
        # Extract configuration from filename
        basename = os.path.basename(edges_file)
        config_name = basename.replace(f"{args.model_name}_", "").replace("_top_edges.txt", "")
        
        print(f"\nProcessing: {config_name}")
        
        # Load edges
        edges = load_top_edges(edges_file, args.top_k)
        print(f"  Loaded {len(edges)} edges")
        
        # Compute distribution
        layer_percentages, layer_counts = compute_layer_distribution(edges, n_layers)
        
        # Print summary
        print(f"  Layer distribution:")
        for layer in sorted(layer_percentages.keys()):
            if layer_percentages[layer] >= args.threshold:
                print(f"    Layer {layer}: {layer_percentages[layer]:.2f}% ({layer_counts[layer]} edges)")
        
        # Plot
        plot_file = os.path.join(args.output_dir, f"{args.model_name}_{config_name}_layer_distribution.png")
        plot_layer_distribution(layer_percentages, config_name, plot_file, args.threshold)

    print("\n" + "="*50)
    print("LAYER-WISE DISTRIBUTION ANALYSIS COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()

