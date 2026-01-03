#!/usr/bin/env python3
"""
Stability Analysis Script
Analyzes stability of important edges across different variations:
- Grammatical structures (DSS1 vs DSS2, GSS1 vs GSS2)
- Bias types (demographic vs gender)
- Different configurations

Usage:
    python stability_analysis.py <model_name> <results_dir> [--HF_token TOKEN] [--output_dir DIR] [--top_k K]

Arguments:
    model_name: Model name (e.g., 'gpt2', 'gpt2-large')
    results_dir: Directory containing circuit finder results (with top_edges.txt files)
    --HF_token: Hugging Face token (optional, for gated models)
    --output_dir: Output directory for stability analysis results (default: current directory)
    --top_k: Number of top edges to consider for overlap (default: 100)
"""

import sys
import os
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_top_edges(edges_file, top_k=100):
    """Load top K edges from a results file."""
    edges = []
    try:
        with open(edges_file, 'r') as f:
            lines = f.readlines()
            for line in lines[2:]:  # Skip header lines
                if line.strip() and ':' in line:
                    try:
                        edge_id = line.split(':')[0].split('.')[1].strip()
                        edges.append(edge_id)
                        if len(edges) >= top_k:
                            break
                    except:
                        continue
    except Exception as e:
        print(f"Warning: Could not load edges from {edges_file}: {e}")
    return set(edges[:top_k])


def compute_overlap(edges1, edges2):
    """Compute overlap percentage between two edge sets."""
    if len(edges1) == 0 and len(edges2) == 0:
        return 100.0
    if len(edges1) == 0 or len(edges2) == 0:
        return 0.0
    intersection = len(edges1.intersection(edges2))
    union = len(edges1.union(edges2))
    if union == 0:
        return 0.0
    return (intersection / union) * 100.0


def find_result_files(results_dir, model_name):
    """Find all top_edges.txt files in results directory."""
    pattern = os.path.join(results_dir, f"{model_name}_*_*_*_top_edges.txt")
    files = glob.glob(pattern)
    
    # Parse file names to extract configuration
    configs = {}
    for file in files:
        basename = os.path.basename(file)
        parts = basename.replace(f"{model_name}_", "").replace("_top_edges.txt", "").split("_")
        if len(parts) >= 3:
            bias_type = parts[0]
            sen = parts[1]
            metric = parts[2]
            key = f"{bias_type}_{sen}_{metric}"
            configs[key] = file
    
    return configs


def create_overlap_matrix(configs, top_k=100):
    """Create overlap matrix between all configurations."""
    # Load edges for each configuration
    edge_sets = {}
    for key, file in configs.items():
        edge_sets[key] = load_top_edges(file, top_k)
    
    # Create matrix
    keys = sorted(edge_sets.keys())
    n = len(keys)
    matrix = np.zeros((n, n))
    
    for i, key1 in enumerate(keys):
        for j, key2 in enumerate(keys):
            if i == j:
                matrix[i, j] = 100.0
            else:
                matrix[i, j] = compute_overlap(edge_sets[key1], edge_sets[key2])
    
    return matrix, keys


def plot_overlap_matrix(matrix, labels, output_file, title="Edge Overlap Matrix"):
    """Plot overlap matrix as heatmap."""
    plt.figure(figsize=(max(10, len(labels) * 0.8), max(8, len(labels) * 0.7)))
    sns.heatmap(matrix, annot=True, fmt='.1f', cmap='YlOrRd', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Overlap %'}, vmin=0, vmax=100)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Configuration', fontsize=12)
    plt.ylabel('Configuration', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Overlap matrix saved to: {output_file}")


def analyze_grammatical_stability(configs, top_k=100):
    """Analyze stability across grammatical structures."""
    results = {}
    
    # Group by bias type and metric
    for key, file in configs.items():
        parts = key.split('_')
        if len(parts) >= 3:
            bias_type = parts[0]
            sen = parts[1]
            metric = parts[2]
            
            group_key = f"{bias_type}_{metric}"
            if group_key not in results:
                results[group_key] = {}
            results[group_key][sen] = load_top_edges(file, top_k)
    
    # Compute overlaps
    overlaps = {}
    for group_key, sens in results.items():
        if len(sens) >= 2:
            # Try DSS1 vs DSS2 or GSS1 vs GSS2
            if 'DSS1' in sens and 'DSS2' in sens:
                overlaps[f"{group_key}_DSS1_vs_DSS2"] = compute_overlap(
                    sens['DSS1'], sens['DSS2']
                )
            if 'GSS1' in sens and 'GSS2' in sens:
                overlaps[f"{group_key}_GSS1_vs_GSS2"] = compute_overlap(
                    sens['GSS1'], sens['GSS2']
                )
    
    return overlaps


def analyze_bias_type_stability(configs, top_k=100):
    """Analyze stability across different bias types."""
    # Group by sentence structure and metric
    results = {}
    
    for key, file in configs.items():
        parts = key.split('_')
        if len(parts) >= 3:
            bias_type = parts[0]
            sen = parts[1]
            metric = parts[2]
            
            group_key = f"{sen}_{metric}"
            if group_key not in results:
                results[group_key] = {}
            results[group_key][bias_type] = load_top_edges(file, top_k)
    
    # Compute overlaps between demographic and gender
    overlaps = {}
    for group_key, bias_types in results.items():
        if 'demographic' in bias_types and 'gender' in bias_types:
            overlaps[f"{group_key}_demographic_vs_gender"] = compute_overlap(
                bias_types['demographic'], bias_types['gender']
            )
    
    return overlaps


def main():
    parser = argparse.ArgumentParser(
        description="Stability Analysis of Bias Circuits"
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
        "--HF_token",
        type=str,
        default=None,
        help="Hugging Face token (not used but kept for consistency)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for stability analysis results (default: current directory)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Number of top edges to consider for overlap (default: 100)"
    )

    args = parser.parse_args()

    # Find all result files
    print(f"Searching for result files in: {args.results_dir}")
    configs = find_result_files(args.results_dir, args.model_name)
    
    if len(configs) == 0:
        print(f"Error: No result files found in {args.results_dir}")
        print(f"Expected files matching pattern: {args.model_name}_*_*_*_top_edges.txt")
        return
    
    print(f"Found {len(configs)} configurations:")
    for key in sorted(configs.keys()):
        print(f"  - {key}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Create full overlap matrix
    print("\nComputing overlap matrix...")
    matrix, labels = create_overlap_matrix(configs, args.top_k)
    
    # Save matrix as CSV
    matrix_file = os.path.join(args.output_dir, f"{args.model_name}_overlap_matrix.csv")
    df_matrix = pd.DataFrame(matrix, index=labels, columns=labels)
    df_matrix.to_csv(matrix_file)
    print(f"Overlap matrix saved to: {matrix_file}")
    
    # Plot overlap matrix
    plot_file = os.path.join(args.output_dir, f"{args.model_name}_overlap_matrix.png")
    plot_overlap_matrix(matrix, labels, plot_file, 
                       title=f"Edge Overlap Matrix - {args.model_name}")

    # Analyze grammatical stability
    print("\nAnalyzing grammatical structure stability...")
    grammatical_overlaps = analyze_grammatical_stability(configs, args.top_k)
    
    if grammatical_overlaps:
        print("Grammatical Structure Overlaps:")
        for key, overlap in grammatical_overlaps.items():
            print(f"  {key}: {overlap:.2f}%")
        
        # Save to file
        gram_file = os.path.join(args.output_dir, f"{args.model_name}_grammatical_stability.txt")
        with open(gram_file, 'w') as f:
            f.write("Grammatical Structure Stability Analysis\n")
            f.write("="*80 + "\n\n")
            for key, overlap in grammatical_overlaps.items():
                f.write(f"{key}: {overlap:.2f}%\n")
        print(f"Grammatical stability saved to: {gram_file}")

    # Analyze bias type stability
    print("\nAnalyzing bias type stability...")
    bias_overlaps = analyze_bias_type_stability(configs, args.top_k)
    
    if bias_overlaps:
        print("Bias Type Overlaps:")
        for key, overlap in bias_overlaps.items():
            print(f"  {key}: {overlap:.2f}%")
        
        # Save to file
        bias_file = os.path.join(args.output_dir, f"{args.model_name}_bias_type_stability.txt")
        with open(bias_file, 'w') as f:
            f.write("Bias Type Stability Analysis\n")
            f.write("="*80 + "\n\n")
            for key, overlap in bias_overlaps.items():
                f.write(f"{key}: {overlap:.2f}%\n")
        print(f"Bias type stability saved to: {bias_file}")

    print("\n" + "="*50)
    print("STABILITY ANALYSIS COMPLETE")
    print("="*50)


if __name__ == "__main__":
    import torch
    main()

