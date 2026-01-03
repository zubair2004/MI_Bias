#!/usr/bin/env python3
"""
Unified Bias Circuit Finder
Finds bias circuits in language models using Edge Attribution Patching (EAP).

Usage:
    python bias_circuit_finder.py <bias_type> <sen> <model_name> <metric> <dataset_file> [--HF_token TOKEN] [--n_edges N] [--output_dir DIR]

Arguments:
    bias_type: 'gender' or 'demographic'
    sen: Sentence template - 'GSS1', 'GSS2', 'DSS1', or 'DSS2'
    model_name: Model name (e.g., 'gpt2', 'gpt2-large')
    metric: 'M1' or 'M2' (M1: prob_diff, M2: prob_diff_new)
    dataset_file: Path to dataset CSV file (clean,corrupted columns)
    --HF_token: Hugging Face token (optional, for gated models)
    --n_edges: Number of top edges to include in circuit (default: 30)
    --output_dir: Output directory for results (default: current directory)
"""

import sys
import os
import argparse
import torch
import numpy as np
import pandas as pd
from functools import partial
from tqdm import tqdm
from huggingface_hub import HfFolder
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
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
    df = pd.read_csv(dataset_file)
    if 'clean' not in df.columns or 'corrupted' not in df.columns:
        raise ValueError(f"Dataset file must contain 'clean' and 'corrupted' columns. Found: {df.columns.tolist()}")
    
    dataset = []
    for _, row in df.iterrows():
        dataset.append(([row['clean']], [row['corrupted']]))
    return dataset




def main():
    parser = argparse.ArgumentParser(
        description="Unified Bias Circuit Finder using Edge Attribution Patching"
    )
    parser.add_argument(
        "bias_type",
        choices=['gender', 'demographic'],
        help="Type of bias: 'gender' or 'demographic'"
    )
    parser.add_argument(
        "sen",
        choices=['GSS1', 'GSS2', 'DSS1', 'DSS2'],
        help="Sentence template: GSS1/GSS2 for gender, DSS1/DSS2 for demographic"
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Model name (e.g., 'gpt2', 'gpt2-large')"
    )
    parser.add_argument(
        "metric",
        choices=['M1', 'M2'],
        help="Metric: M1 (prob_diff) or M2 (prob_diff_new)"
    )
    parser.add_argument(
        "dataset_file",
        type=str,
        help="Path to dataset CSV file with 'clean' and 'corrupted' columns"
    )
    parser.add_argument(
        "--HF_token",
        type=str,
        default=None,
        help="Hugging Face token (required for gated models)"
    )
    parser.add_argument(
        "--n_edges",
        type=int,
        default=30,
        help="Number of top edges to include in circuit (default: 30)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for results (default: current directory)"
    )

    args = parser.parse_args()

    # Validate sentence template matches bias type
    if args.bias_type == 'gender' and args.sen not in ['GSS1', 'GSS2']:
        parser.error(f"Gender bias requires GSS1 or GSS2, got {args.sen}")
    if args.bias_type == 'demographic' and args.sen not in ['DSS1', 'DSS2']:
        parser.error(f"Demographic bias requires DSS1 or DSS2, got {args.sen}")

    # Setup Hugging Face token if provided
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
    if not os.path.exists(args.dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_file}")
    dataset = load_dataset_from_csv(args.dataset_file)
    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty: {args.dataset_file}")
    print(f"Loaded {len(dataset)} samples")

    # Create computational graph
    print("Creating computational graph...")
    g = Graph.from_model(model)

    print(f'Total No. of Nodes in Model: {len(g.nodes)}')
    print(f'Total No. of Edges in Model: {len(g.edges)}')

    # Select metric
    if args.metric == 'M1':
        metric_fn = partial(prob_diff, model)
    elif args.metric == 'M2':
        metric_fn = partial(prob_diff_new, model)
    else:
        raise ValueError(f"Unknown metric: {args.metric}")

    # Baseline scoring
    print("\n" + "="*50)
    print("BASELINE SCORING")
    print("="*50)
    baseline_score = evaluate_baseline(model, dataset, metric_fn).mean()
    print(f"Baseline performance: {baseline_score}")

    # Graph baseline scoring
    graph_baseline = evaluate_graph(model, g, dataset, metric_fn).mean()
    print(f"Graph baseline performance: {graph_baseline}")

    # Edge Attribution Patching
    print("\n" + "="*50)
    print("EDGE ATTRIBUTION PATCHING (EAP)")
    print("="*50)
    print("Computing edge attribution scores...")
    attribute(model, g, dataset, partial(metric_fn, loss=True, mean=True))

    # Get top scoring edges
    scores = g.scores(absolute=True)
    print(f"Computed scores for {len(scores)} edges")

    # Apply greedy algorithm to select top edges
    print(f"Selecting top {args.n_edges} edges using greedy algorithm...")
    g.apply_greedy(args.n_edges)
    g.prune_dead_nodes()

    print(f"Circuit contains {g.count_included_edges()} edges and {g.count_included_nodes()} nodes")

    # Evaluate circuit performance
    circuit_score = evaluate_graph(model, g, dataset, metric_fn).mean()
    print(f"\nCircuit performance: {circuit_score}")
    print(f"Score difference (baseline - circuit): {baseline_score - circuit_score}")

    # Save circuit visualization
    os.makedirs(args.output_dir, exist_ok=True)
    output_prefix = f"{args.model_name}_{args.bias_type}_{args.sen}_{args.metric}"
    
    try:
        gz = g.to_graphviz()
        viz_file = os.path.join(args.output_dir, f"{output_prefix}_circuit.png")
        gz.draw(viz_file, prog='dot')
        print(f"Circuit visualization saved to: {viz_file}")
    except Exception as e:
        print(f"Warning: Could not generate circuit visualization: {e}")

    # Save top edges
    remaining_edges = list(g.edges.items())
    remaining_edges.sort(key=lambda x: abs(x[1].score) if x[1].score is not None else 0, reverse=True)
    
    edges_file = os.path.join(args.output_dir, f"{output_prefix}_top_edges.txt")
    with open(edges_file, "w") as f:
        f.write(f"Top {args.n_edges} edges for {args.model_name} {args.bias_type} {args.sen} {args.metric}\n")
        f.write("="*80 + "\n")
        for i, (edge_id, edge) in enumerate(remaining_edges[:args.n_edges]):
            score = edge.score if edge.score is not None else 0.0
            f.write(f"{i+1}. {edge_id}: score={score:.6f}\n")
    print(f"Top edges saved to: {edges_file}")

    # Save results summary
    results_file = os.path.join(args.output_dir, f"{output_prefix}_results.txt")
    with open(results_file, "w") as f:
        f.write(f"Bias Circuit Analysis Results\n")
        f.write("="*80 + "\n")
        f.write(f"Bias Type: {args.bias_type}\n")
        f.write(f"Sentence Template: {args.sen}\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Metric: {args.metric}\n")
        f.write(f"Dataset: {args.dataset_file}\n")
        f.write(f"Number of samples: {len(dataset)}\n")
        f.write(f"\nResults:\n")
        f.write(f"  Baseline Score: {baseline_score:.6f}\n")
        f.write(f"  Graph Baseline Score: {graph_baseline:.6f}\n")
        f.write(f"  Circuit Score: {circuit_score:.6f}\n")
        f.write(f"  Score Difference: {baseline_score - circuit_score:.6f}\n")
        f.write(f"\nCircuit Statistics:\n")
        f.write(f"  Number of edges in circuit: {g.count_included_edges()}\n")
        f.write(f"  Number of nodes in circuit: {g.count_included_nodes()}\n")
    print(f"Results summary saved to: {results_file}")

    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()

