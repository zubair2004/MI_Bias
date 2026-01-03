#!/usr/bin/env python3
"""
Ablation Analysis Visualization
Visualizes the drop in metric value as edges are progressively ablated.

Usage:
    python Plots/ablation_analysis.py <model_name> <bias_type> <sen> <metric> <dataset_file> [--HF_token TOKEN] [--output_dir DIR] [--max_edges N] [--intervals LIST]

Arguments:
    model_name: Model name (e.g., 'gpt2', 'gpt2-large')
    bias_type: 'gender' or 'demographic'
    sen: Sentence template - 'GSS1', 'GSS2', 'DSS1', or 'DSS2'
    metric: 'M1' or 'M2'
    dataset_file: Path to dataset CSV file
    --HF_token: Hugging Face token (optional, for gated models)
    --output_dir: Output directory for plots (default: current directory)
    --max_edges: Maximum number of edges to ablate (default: 1000)
    --intervals: Comma-separated list of edge counts for ablation (default: 10,50,100,150,200,250,300,350,400,450,500)
"""

import sys
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
from huggingface_hub import HfFolder
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Helper_Functions.graph import Graph
from Helper_Functions.attribution import attribute
from Metrics.prob_diff import prob_diff
from Metrics.prob_diff_new import prob_diff_new
from Metrics.evaluate_baseline import evaluate_baseline
from Metrics.evaluate_graph import evaluate_graph

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_dataset_from_csv(dataset_file):
    """Load dataset from CSV file with clean and corrupted columns."""
    import pandas as pd
    df = pd.read_csv(dataset_file)
    if 'clean' not in df.columns or 'corrupted' not in df.columns:
        raise ValueError(f"Dataset file must contain 'clean' and 'corrupted' columns.")
    
    dataset = []
    for _, row in df.iterrows():
        dataset.append(([row['clean']], [row['corrupted']]))
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Ablation Analysis Visualization"
    )
    parser.add_argument("model_name", type=str, help="Model name")
    parser.add_argument("bias_type", choices=['gender', 'demographic'], help="Type of bias")
    parser.add_argument("sen", choices=['GSS1', 'GSS2', 'DSS1', 'DSS2'], help="Sentence template")
    parser.add_argument("metric", choices=['M1', 'M2'], help="Metric: M1 or M2")
    parser.add_argument("dataset_file", type=str, help="Path to dataset CSV file")
    parser.add_argument("--HF_token", type=str, default=None, help="Hugging Face token")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory")
    parser.add_argument("--max_edges", type=int, default=1000, help="Maximum edges to ablate")
    parser.add_argument("--intervals", type=str, default="10,50,100,150,200,250,300,350,400,450,500",
                       help="Comma-separated edge counts for ablation")

    args = parser.parse_args()

    if args.HF_token:
        HfFolder.save_token(args.HF_token)

    # Load model
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_bos_token = False

    model = HookedTransformer.from_pretrained(
        args.model_name,
        device=device,
        tokenizer=tokenizer
    )
    model.cfg.use_attn_in = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True

    # Load dataset
    print(f"Loading dataset from: {args.dataset_file}")
    dataset = load_dataset_from_csv(args.dataset_file)

    # Select metric
    if args.metric == 'M1':
        metric_fn = partial(prob_diff, model)
    else:
        metric_fn = partial(prob_diff_new, model)

    # Create graph and compute attributions
    print("Creating computational graph and computing edge attributions...")
    g = Graph.from_model(model)
    
    # Baseline
    baseline_score = evaluate_baseline(model, dataset, metric_fn).mean().item()
    graph_baseline = evaluate_graph(model, g, dataset, metric_fn).mean().item()
    
    print(f"Baseline score: {baseline_score:.6f}")
    print(f"Graph baseline score: {graph_baseline:.6f}")

    # Compute edge attributions
    attribute(model, g, dataset, partial(metric_fn, loss=True, mean=True))

    # Get sorted edges
    remaining_edges = list(g.edges.items())
    remaining_edges.sort(key=lambda x: abs(x[1].score) if x[1].score is not None else 0, reverse=True)
    
    # Parse intervals
    intervals = [int(x.strip()) for x in args.intervals.split(',')]
    intervals = [0] + [i for i in intervals if i <= args.max_edges]
    
    # Perform ablation
    print(f"\nPerforming ablation at intervals: {intervals}")
    ablation_scores = [graph_baseline]
    
    for n_edges in intervals[1:]:
        print(f"  Ablating top {n_edges} edges...")
        
        # Reset graph
        g_ablated = Graph.from_model(model)
        attribute(model, g_ablated, dataset, partial(metric_fn, loss=True, mean=True))
        
        # Get sorted edges
        edges_sorted = list(g_ablated.edges.items())
        edges_sorted.sort(key=lambda x: abs(x[1].score) if x[1].score is not None else 0, reverse=True)
        
        # Ablate top n edges
        for i in range(min(n_edges, len(edges_sorted))):
            edge_id = edges_sorted[i][0]
            g_ablated.edges[edge_id].in_graph = False
        
        g_ablated.prune_dead_nodes()
        
        # Evaluate
        score = evaluate_graph(model, g_ablated, dataset, metric_fn).mean().item()
        ablation_scores.append(score)
        print(f"    Score after ablation: {score:.6f}")

    # Plot results
    os.makedirs(args.output_dir, exist_ok=True)
    output_prefix = f"{args.model_name}_{args.bias_type}_{args.sen}_{args.metric}"
    
    plt.figure(figsize=(12, 6))
    plt.plot(intervals, ablation_scores, marker='o', color='green', linewidth=2, markersize=8, label='Ablated Circuit')
    plt.axhline(y=baseline_score, color='red', linestyle='--', linewidth=2, label='Baseline')
    plt.axhline(y=graph_baseline, color='blue', linestyle='--', linewidth=2, label='Graph Baseline')
    plt.xlabel('Number of Top Edges Ablated', fontsize=14, fontweight='bold')
    plt.ylabel('Metric Value', fontsize=14, fontweight='bold')
    plt.title(f'Ablation Analysis: {args.model_name} - {args.bias_type} {args.sen} {args.metric}', 
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    plot_file = os.path.join(args.output_dir, f"{output_prefix}_ablation.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nAblation plot saved to: {plot_file}")
    
    # Save data
    data_file = os.path.join(args.output_dir, f"{output_prefix}_ablation_data.txt")
    with open(data_file, 'w') as f:
        f.write(f"Ablation Analysis Results\n")
        f.write("="*80 + "\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Bias Type: {args.bias_type}\n")
        f.write(f"Sentence: {args.sen}\n")
        f.write(f"Metric: {args.metric}\n")
        f.write(f"\nBaseline Score: {baseline_score:.6f}\n")
        f.write(f"Graph Baseline Score: {graph_baseline:.6f}\n")
        f.write(f"\nAblation Results:\n")
        f.write(f"{'Edges Ablated':<20} {'Score':<20}\n")
        f.write("-" * 40 + "\n")
        for n, score in zip(intervals, ablation_scores):
            f.write(f"{n:<20} {score:<20.6f}\n")
    print(f"Ablation data saved to: {data_file}")

    print("\n" + "="*50)
    print("ABLATION ANALYSIS COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()

