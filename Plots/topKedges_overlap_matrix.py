#!/usr/bin/env python3
"""
Top-K Edges Overlap Matrix
Creates overlap matrices for top-K edges across different configurations.
This is a wrapper that calls stability_analysis.py with visualization focus.

Usage:
    python Plots/topKedges_overlap_matrix.py <model_name> <results_dir> [--output_dir DIR] [--top_k K]
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import and run stability analysis
from Bias_Code.stability_analysis import main

if __name__ == "__main__":
    main()
