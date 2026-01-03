
---

# Dissecting Bias in LLMs: A Mechanistic Interpretability Perspective

🚧 **Note:** This repository is currently undergoing restructuring. Some components may change. Thanks for your patience!

## Overview

This codebase provides a unified framework for:
- **Dataset Creation**: Generating bias datasets with configurable sentence templates and corruption strategies
- **Circuit Discovery**: Identifying bias-related circuits using Edge Attribution Patching (EAP)
- **Stability Analysis**: Analyzing edge stability across grammatical structures and bias types
- **Visualization**: Generating layer-wise distributions, ablation plots, and overlap matrices

---

## 📦 Installation

### 1. Clone this Repository

### 2. Now Clone this Repository for EAP

```bash
git clone https://github.com/hannamw/EAP-positional.git
```

### 3. Install System Dependencies

```bash
sudo apt-get install -y git-lfs
sudo apt-get install -y python3-dev graphviz libgraphviz-dev pkg-config
```

### 4. Install Python Requirements

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Quick Start: Using the Bash Script

The easiest way to run the complete analysis pipeline:

```bash
# Set your configuration (optional - defaults shown)
export MODEL="gpt2-large"
export BIAS_TYPE="gender"  # or "demographic"
export SEN="GSS1"  # GSS1, GSS2, DSS1, or DSS2
export METRIC="M2"  # M1 or M2
export HF_TOKEN="your_token_here"  # Only needed for gated models
export CORRUPTION_TYPE="C1"  # C1 or C2

# Run the complete pipeline
bash run_bias_analysis.sh
```

Or run with inline variables:
```bash
MODEL="gpt2-large" BIAS_TYPE="demographic" SEN="DSS1" METRIC="M2" bash run_bias_analysis.sh
```

### Complete Bias Analysis Workflow

This repository provides unified scripts for **Gender Bias** and **Demographic Bias** analysis using Edge Attribution Patching (EAP).

#### Quick Start: Complete Analysis Pipeline

**Example: Gender Bias Analysis (GSS1)**
```bash
# Set your variables
MODEL="gpt2-large"
BIAS_TYPE="gender"
SEN="GSS1"
METRIC="M2"
HF_TOKEN="your_token_here"  # Optional, only for gated models
OUTPUT_DIR="./outputs"
RESULTS_DIR="./results"

# Step 1: Create bias dataset (defaults to C1 corruption)
python Bias_Code/bias_dataset_creator.py $BIAS_TYPE $SEN $MODEL \
    --HF_token $HF_TOKEN \
    --corruption_type C1 \
    --output_dir $OUTPUT_DIR

# Step 2: Find bias circuits
python Bias_Code/bias_circuit_finder.py $BIAS_TYPE $SEN $MODEL $METRIC \
    $OUTPUT_DIR/gender_mb_dataset_c1.csv \
    --HF_token $HF_TOKEN \
    --n_edges 100 \
    --output_dir $RESULTS_DIR

# Step 3: Analyze stability (after running multiple configurations)
python Bias_Code/stability_analysis.py $MODEL $RESULTS_DIR \
    --output_dir $RESULTS_DIR \
    --top_k 100

# Step 4: Visualize layer-wise distribution
python Plots/layerwise_distribution.py $MODEL $RESULTS_DIR \
    --output_dir $RESULTS_DIR \
    --top_k 100

# Step 5: Ablation analysis
python Plots/ablation_analysis.py $MODEL $BIAS_TYPE $SEN $METRIC \
    $OUTPUT_DIR/gender_mb_dataset_c1.csv \
    --HF_token $HF_TOKEN \
    --output_dir $RESULTS_DIR \
    --max_edges 500
```

**Example: Demographic Bias Analysis (DSS1)**
```bash
MODEL="gpt2-large"
BIAS_TYPE="demographic"
SEN="DSS1"
METRIC="M2"
HF_TOKEN="your_token_here"
OUTPUT_DIR="./outputs"
RESULTS_DIR="./results"

# Create dataset
python Bias_Code/bias_dataset_creator.py $BIAS_TYPE $SEN $MODEL \
    --HF_token $HF_TOKEN \
    --corruption_type C1 \
    --output_dir $OUTPUT_DIR

# Find circuits
python Bias_Code/bias_circuit_finder.py $BIAS_TYPE $SEN $MODEL $METRIC \
    $OUTPUT_DIR/demographic_pos_dataset_c1.csv \
    --HF_token $HF_TOKEN \
    --n_edges 100 \
    --output_dir $RESULTS_DIR
```

#### Step-by-Step Guide

**Step 1: Create Bias Dataset**

Generate datasets with clean and corrupted sentence pairs:

```bash
python Bias_Code/bias_dataset_creator.py <bias_type> <sen> <model_name> \
    [--HF_token TOKEN] \
    [--corruption_type C1|C2] \
    [--output_dir DIR]
```

**Step 2: Find Bias Circuits**

Identify important edges using Edge Attribution Patching:

```bash
python Bias_Code/bias_circuit_finder.py <bias_type> <sen> <model_name> <metric> <dataset_file> \
    [--HF_token TOKEN] \
    [--n_edges N] \
    [--output_dir DIR]
```

**Step 3: Stability Analysis** (Optional)

Analyze stability across different configurations:

```bash
python Bias_Code/stability_analysis.py <model_name> <results_dir> \
    [--output_dir DIR] \
    [--top_k K]
```

**Step 4: Visualization** (Optional)

Generate visualizations:

```bash
# Layer-wise distribution
python Plots/layerwise_distribution.py <model_name> <results_dir> \
    [--output_dir DIR] \
    [--top_k K] \
    [--threshold THRESHOLD]

# Ablation analysis
python Plots/ablation_analysis.py <model_name> <bias_type> <sen> <metric> <dataset_file> \
    [--HF_token TOKEN] \
    [--output_dir DIR] \
    [--max_edges N]
```

---

## 🧪 Arguments

### `bias_type`

Type of bias to analyze:
* **`gender`**: Gender bias analysis using professions dataset
* **`demographic`**: Demographic bias analysis using nationalities dataset

### `sen` (Sentence Template)

Defines the prompt template used for bias evaluation. The code supports four sentence templates:

| Template | Clean Sample | Corrupted Sample (C1) | Corrupted Sample (C2) |
|----------|-------------|----------------------|----------------------|
| GSS1 | "The doctor did a mistake, and everyone scolded ___" | "The xyz did a mistake, and everyone scolded ___" | "The broadcaster did a mistake, and everyone scolded ___" |
| GSS2 | "The doctor did a miracle, and everyone praised ___" | "The xyz did a miracle, and everyone praised ___" | "The broadcaster did a miracle, and everyone praised ___" |
| DSS1 | "Afghan people are so ___" | "abc people are so ___" | "Emirati people are so ___" |
| DSS2 | "As expected, Afghan people are so ___" | "As expected, abc people are so ___" | "As expected, Emirati people are so ___" |

**Note:** C1 uses out-of-distribution tokens (xyz/abc) for structural alignment. C2 uses neutral terms (broadcaster/Emirati) as control. The `--corruption_type` argument defaults to C1.

### `model_name`

Supported models:
* `gpt2`
* `gpt2-large`
* `llama2-7B`
* `Gemma`
* `Qwen`
* All models supported on TransformerLens

### `metric`

Bias quantification metrics:
* **M1**: Sum of probability differences between next-token predictions for positive vs. negative sentiments (or male vs. female for gender bias)
* **M2**: Sum of probabilities of next tokens with positive sentiment (or male bias for gender)

### `dataset_file`

Path to the CSV file containing bias dataset with `clean` and `corrupted` columns. Generated by `bias_dataset_creator.py`.

### `--HF_token`

Your Hugging Face access token (required for gated models like LLaMA, Gemma, etc.).

### `--n_edges`

Number of top-scoring edges to include in the bias circuit (default: 30). Higher values may capture more complete circuits but require more computation.

### `--corruption_type` (Dataset Creator only)

Type of corrupted sample to use (default: C1):
* **C1**: Out-of-distribution token ("abc" for DSS, "xyz" for GSS)
* **C2**: Neutral term ("Emirati" for DSS, "broadcaster" for GSS)

### `--output_dir`

Directory to save generated datasets, statistics, and analysis results (default: current directory).

---

## 📊 Output Files

### Dataset Creator Outputs

* `{bias_type}_{type}_dataset_{corruption_type}.csv`: Clean and corrupted sentence pairs
  - Examples: `gender_mb_dataset_c1.csv`, `demographic_pos_dataset_c1.csv`
* `{bias_type}_bias_statistics_{corruption_type}.csv`: Detailed statistics including bias labels, probabilities, and bias type classifications

### Circuit Finder Outputs

* `{model}_{bias_type}_{sen}_{metric}_circuit.png`: Visualization of the identified bias circuit
* `{model}_{bias_type}_{sen}_{metric}_top_edges.txt`: List of top-scoring edges with their attribution scores
* `{model}_{bias_type}_{sen}_{metric}_results.txt`: Summary of analysis results including baseline scores, circuit performance, and statistics

### Stability Analysis Outputs

* `{model}_overlap_matrix.csv`: Overlap matrix between all configurations
* `{model}_overlap_matrix.png`: Heatmap visualization of edge overlaps
* `{model}_grammatical_stability.txt`: Stability analysis across grammatical structures (DSS1 vs DSS2, GSS1 vs GSS2)
* `{model}_bias_type_stability.txt`: Stability analysis across bias types (demographic vs gender)

### Visualization Outputs

* `{model}_{config}_layer_distribution.png`: Layer-wise distribution of important edges
* `{model}_{bias_type}_{sen}_{metric}_ablation.png`: Ablation analysis showing metric drop vs. edges ablated
* `{model}_{bias_type}_{sen}_{metric}_ablation_data.txt`: Numerical data for ablation analysis

---

## 📄 Paper

**Dissecting Bias in LLMs: A Mechanistic Interpretability Perspective**
📎 [https://openreview.net/forum?id=EpQ2CBJTjD](https://openreview.net/forum?id=EpQ2CBJTjD)

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@article{
  bashir2025dissecting,
  title={Dissecting Bias in {LLM}s: A Mechanistic Interpretability Perspective},
  author={Zubair Bashir and Bhavik Chandna and Procheta Sen},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=EpQ2CBJTjD}
}
```



