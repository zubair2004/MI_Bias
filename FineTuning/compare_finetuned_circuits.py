#!/usr/bin/env python3
"""
Compare Circuits: Pretrained vs Fine-tuned Models
Compares bias circuits between pretrained and fine-tuned models.

Usage:
    python FineTuning/compare_finetuned_circuits.py <base_model_name> <pretrained_results_dir> <finetuned_model_path> <bias_type> <sen> <metric> <dataset_file> [--HF_token TOKEN] [--output_dir DIR] [--n_edges N]

Arguments:
    base_model_name: Base model name (e.g., 'gpt2', 'gpt2-large')
    pretrained_results_dir: Directory containing pretrained model circuit results
    finetuned_model_path: Path to fine-tuned model directory
    bias_type: 'gender' or 'demographic'
    sen: Sentence template - 'GSS1', 'GSS2', 'DSS1', or 'DSS2'
    metric: 'M1' or 'M2'
    dataset_file: Path to dataset CSV file
    --HF_token: Hugging Face token (optional, for gated models)
    --output_dir: Output directory for comparison results (default: current directory)
    --n_edges: Number of top edges to compare (default: 100)
"""

import sys
import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from huggingface_hub import HfFolder
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Helper_Functions.graph import Graph
from Helper_Functions.attribution import attribute
from Metrics.prob_diff import prob_diff
from Metrics.prob_diff_new import prob_diff_new

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_top_edges_from_file(edges_file, top_k=100):
    """Load top K edges from a results file."""
    edges = []
    try:
        with open(edges_file, 'r') as f:
            lines = f.readlines()
            for line in lines[2:]:  # Skip header lines
                if line.strip() and ':' in line:
                    try:
                        parts = line.split(':')
                        if len(parts) >= 1:
                            edge_part = parts[0].strip()
                            if '.' in edge_part:
                                edge_id = edge_part.split('.', 1)[1].strip()
                                if edge_id:
                                    edges.append(edge_id)
                                    if len(edges) >= top_k:
                                        break
                    except (IndexError, ValueError):
                        continue
    except Exception as e:
        print(f"Warning: Could not load edges from {edges_file}: {e}")
    return set(edges[:top_k])


def find_circuit_results(results_dir, model_name, bias_type, sen, metric):
    """Find circuit results file for given configuration."""
    pattern = os.path.join(results_dir, f"{model_name}_{bias_type}_{sen}_{metric}_top_edges.txt")
    if os.path.exists(pattern):
        return pattern
    
    # Try alternative patterns
    patterns = [
        os.path.join(results_dir, f"**/{model_name}_{bias_type}_{sen}_{metric}_top_edges.txt"),
        os.path.join(results_dir, f"{model_name}_*_{bias_type}_{sen}_{metric}_top_edges.txt"),
    ]
    
    import glob
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            return files[0]
    
    return None


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


def find_circuit_for_model(model, model_name, bias_type, sen, metric, dataset_file, n_edges=100, hf_token=None):
    """Find bias circuit for a given model."""
    print(f"\nFinding circuit for {model_name}...")
    
    # Select metric
    if metric == 'M1':
        metric_fn = partial(prob_diff, model)
    else:
        metric_fn = partial(prob_diff_new, model)
    
    # Load dataset
    import pandas as pd
    df = pd.read_csv(dataset_file)
    dataset = []
    for _, row in df.iterrows():
        dataset.append(([row['clean']], [row['corrupted']]))
    
    # Create graph and compute attributions
    g = Graph.from_model(model)
    attribute(model, g, dataset, partial(metric_fn, loss=True, mean=True))
    
    # Get top edges
    remaining_edges = list(g.edges.items())
    remaining_edges.sort(key=lambda x: abs(x[1].score) if x[1].score is not None else 0, reverse=True)
    
    top_edges = set([edge_id for edge_id, _ in remaining_edges[:n_edges]])
    
    return top_edges, g


def main():
    parser = argparse.ArgumentParser(
        description="Compare Circuits: Pretrained vs Fine-tuned Models"
    )
    parser.add_argument("base_model_name", type=str, help="Base model name")
    parser.add_argument("pretrained_results_dir", type=str, help="Directory with pretrained model results")
    parser.add_argument("finetuned_model_path", type=str, help="Path to fine-tuned model directory")
    parser.add_argument("bias_type", choices=['gender', 'demographic'], help="Type of bias")
    parser.add_argument("sen", choices=['GSS1', 'GSS2', 'DSS1', 'DSS2'], help="Sentence template")
    parser.add_argument("metric", choices=['M1', 'M2'], help="Metric: M1 or M2")
    parser.add_argument("dataset_file", type=str, help="Path to dataset CSV file")
    parser.add_argument("--HF_token", type=str, default=None, help="Hugging Face token")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory")
    parser.add_argument("--n_edges", type=int, default=100, help="Number of top edges to compare")

    args = parser.parse_args()

    if args.HF_token:
        HfFolder.save_token(args.HF_token)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load pretrained model edges from results file
    print("Loading pretrained model circuit...")
    pretrained_edges_file = find_circuit_results(
        args.pretrained_results_dir, 
        args.base_model_name, 
        args.bias_type, 
        args.sen, 
        args.metric
    )
    
    if not pretrained_edges_file or not os.path.exists(pretrained_edges_file):
        raise FileNotFoundError(
            f"Pretrained circuit results not found. Expected file matching: "
            f"{args.base_model_name}_{args.bias_type}_{args.sen}_{args.metric}_top_edges.txt"
        )
    
    pretrained_edges = load_top_edges_from_file(pretrained_edges_file, args.n_edges)
    print(f"Loaded {len(pretrained_edges)} edges from pretrained model")

    # Load fine-tuned model and find circuit
    print(f"\nLoading fine-tuned model from: {args.finetuned_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.finetuned_model_path)
    tokenizer.add_bos_token = False

    finetuned_model = HookedTransformer.from_pretrained(
        args.finetuned_model_path,
        device=device,
        tokenizer=tokenizer
    )
    finetuned_model.cfg.use_attn_in = True
    finetuned_model.cfg.use_split_qkv_input = True
    finetuned_model.cfg.use_attn_result = True
    finetuned_model.cfg.use_hook_mlp_in = True

    finetuned_edges, finetuned_graph = find_circuit_for_model(
        finetuned_model,
        f"{args.base_model_name}_finetuned",
        args.bias_type,
        args.sen,
        args.metric,
        args.dataset_file,
        args.n_edges,
        args.HF_token
    )
    print(f"Found {len(finetuned_edges)} edges in fine-tuned model")

    # Compute overlap
    overlap = compute_overlap(pretrained_edges, finetuned_edges)
    print(f"\n{'='*60}")
    print(f"OVERLAP ANALYSIS")
    print(f"{'='*60}")
    print(f"Pretrained model edges: {len(pretrained_edges)}")
    print(f"Fine-tuned model edges: {len(finetuned_edges)}")
    print(f"Overlap: {overlap:.2f}%")
    print(f"Intersection: {len(pretrained_edges.intersection(finetuned_edges))} edges")
    print(f"Union: {len(pretrained_edges.union(finetuned_edges))} edges")

    # Save comparison results
    output_prefix = f"{args.base_model_name}_{args.bias_type}_{args.sen}_{args.metric}_finetuned_comparison"
    
    comparison_file = os.path.join(args.output_dir, f"{output_prefix}.txt")
    with open(comparison_file, 'w') as f:
        f.write("Circuit Comparison: Pretrained vs Fine-tuned Model\n")
        f.write("="*80 + "\n\n")
        f.write(f"Base Model: {args.base_model_name}\n")
        f.write(f"Fine-tuned Model Path: {args.finetuned_model_path}\n")
        f.write(f"Bias Type: {args.bias_type}\n")
        f.write(f"Sentence Template: {args.sen}\n")
        f.write(f"Metric: {args.metric}\n\n")
        f.write(f"Pretrained Model Edges: {len(pretrained_edges)}\n")
        f.write(f"Fine-tuned Model Edges: {len(finetuned_edges)}\n")
        f.write(f"Overlap: {overlap:.2f}%\n")
        f.write(f"Intersection: {len(pretrained_edges.intersection(finetuned_edges))} edges\n")
        f.write(f"Union: {len(pretrained_edges.union(finetuned_edges))} edges\n\n")
        
        f.write("Edges only in Pretrained Model:\n")
        f.write("-"*80 + "\n")
        only_pretrained = pretrained_edges - finetuned_edges
        for edge in sorted(only_pretrained):
            f.write(f"  {edge}\n")
        
        f.write("\nEdges only in Fine-tuned Model:\n")
        f.write("-"*80 + "\n")
        only_finetuned = finetuned_edges - pretrained_edges
        for edge in sorted(only_finetuned):
            f.write(f"  {edge}\n")
        
        f.write("\nCommon Edges:\n")
        f.write("-"*80 + "\n")
        common = pretrained_edges.intersection(finetuned_edges)
        for edge in sorted(common):
            f.write(f"  {edge}\n")
    
    print(f"\nComparison results saved to: {comparison_file}")

    # Create visualization
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = ['Pretrained Only', 'Common', 'Fine-tuned Only']
        counts = [
            len(pretrained_edges - finetuned_edges),
            len(pretrained_edges.intersection(finetuned_edges)),
            len(finetuned_edges - pretrained_edges)
        ]
        
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}\n({count/sum(counts)*100:.1f}%)',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Number of Edges', fontsize=12, fontweight='bold')
        ax.set_title(f'Circuit Comparison: Pretrained vs Fine-tuned\n{args.base_model_name} - {args.bias_type} {args.sen} {args.metric}', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(args.output_dir, f"{output_prefix}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to: {plot_file}")
    except Exception as e:
        print(f"Warning: Could not create visualization: {e}")

    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

