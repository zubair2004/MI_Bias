# Fine-tuning Experiments

This directory contains code for fine-tuning experiments to investigate how bias circuits change when models are fine-tuned on different types of data.

## Overview

We investigate whether important edges (circuit components) change when models are fine-tuned on different types of data:

1. **Positive Bias Fine-tuning**: Fine-tune models on datasets where positive things are said about countries/professions that previously showed negative bias. This debiasing fine-tuning aims to check whether the important edges for generating sentences of the same grammatical structure change.

2. **Shakespeare Fine-tuning**: Fine-tune models on Shakespeare text that has no overlap with bias-related sentences or test topics. The bias doesn't change after this, but we check whether the structures important for bias remain the same.


## Usage

### 1. Fine-tune a Model

#### Positive Bias Fine-tuning

```bash
python FineTuning/finetune_model.py gpt2-large positive_bias ./finetuning/gpt2-large_positive_demographic \
    --bias_type demographic \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --HF_token YOUR_HF_TOKEN
```

For gender bias:
```bash
python FineTuning/finetune_model.py gpt2-large positive_bias ./finetuning/gpt2-large_positive_gender \
    --bias_type gender \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 5e-5
```

#### Shakespeare Fine-tuning

```bash
python FineTuning/finetune_model.py gpt2-large shakespeare ./finetuning/gpt2-large_shakespeare \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 5e-5
```

### 2. Compare Circuits

Compare circuits between pretrained and fine-tuned models:

```bash
python FineTuning/compare_finetuned_circuits.py \
    gpt2-large \
    ./results \
    ./finetuning/gpt2-large_positive_demographic \
    demographic \
    DSS1 \
    M2 \
    ./outputs/demographic_pos_dataset_c1.csv \
    --output_dir ./results \
    --n_edges 100 \
    --HF_token YOUR_HF_TOKEN
```

### 3. Run Complete Pipeline

Use the bash script to run the complete experiment pipeline:

```bash
export BASE_MODEL="gpt2-large"
export BIAS_TYPE="demographic"  # or "gender"
export SEN="DSS1"  # GSS1, GSS2, DSS1, or DSS2
export METRIC="M2"  # M1 or M2
export HF_TOKEN="your_token_here"  # Optional

bash FineTuning/run_finetuning_experiments.sh
```

Or with inline variables:

```bash
BASE_MODEL=gpt2-large BIAS_TYPE=demographic SEN=DSS1 METRIC=M2 \
    HF_TOKEN=your_token bash FineTuning/run_finetuning_experiments.sh
```

## Arguments

### `finetune_model.py`

- **`model_name`** (required): Base model name (e.g., 'gpt2', 'gpt2-large', 'meta-llama/Llama-2-7b-hf')
- **`finetune_type`** (required): Type of fine-tuning - 'positive_bias' or 'shakespeare'
- **`output_dir`** (required): Directory to save fine-tuned model
- **`--HF_token`**: Hugging Face token (required for gated models like Llama-2)
- **`--bias_type`**: 'demographic' or 'gender' (only for positive_bias, default: 'demographic')
- **`--epochs`**: Number of training epochs (default: 3)
- **`--batch_size`**: Batch size for training (default: 4)
- **`--learning_rate`**: Learning rate (default: 5e-5)

### `compare_finetuned_circuits.py`

- **`base_model_name`** (required): Base model name
- **`pretrained_results_dir`** (required): Directory containing pretrained model circuit results
- **`finetuned_model_path`** (required): Path to fine-tuned model directory
- **`bias_type`** (required): 'gender' or 'demographic'
- **`sen`** (required): Sentence template - 'GSS1', 'GSS2', 'DSS1', or 'DSS2'
- **`metric`** (required): 'M1' or 'M2'
- **`dataset_file`** (required): Path to dataset CSV file
- **`--HF_token`**: Hugging Face token (optional, for gated models)
- **`--output_dir`**: Output directory for comparison results (default: current directory)
- **`--n_edges`**: Number of top edges to compare (default: 100)

## Output

### Fine-tuning Output

Fine-tuned models are saved in the specified `output_dir` with:
- `pytorch_model.bin` or model weights
- `config.json`
- `tokenizer.json` and related tokenizer files

### Circuit Comparison Output

Comparison results include:

1. **Text Report** (`*_finetuned_comparison.txt`):
   - Overlap statistics
   - Edges only in pretrained model
   - Edges only in fine-tuned model
   - Common edges

2. **Visualization** (`*_finetuned_comparison.png`):
   - Bar chart showing edge distribution across categories

## Dataset Creation

### Positive Bias Dataset

For **demographic bias**:
- Creates positive sentences about all nationalities from `Dataset/Nationalities.csv`
- Examples: "Afghan people are so wonderful", "As expected, Afghan people are so brilliant"

For **gender bias**:
- Creates male-biased sentences for all professions from `Dataset/Professions.csv`
- Examples: "The doctor did a great job, and everyone praised him"

### Shakespeare Dataset

- Uses curated Shakespeare quotes and text


## Example Workflow

```bash
# 1. Create test dataset
python Bias_Code/bias_dataset_creator.py demographic DSS1 gpt2-large \
    --corruption_type C1 --output_dir ./outputs

# 2. Find pretrained circuit
python Bias_Code/bias_circuit_finder.py demographic DSS1 gpt2-large M2 \
    ./outputs/demographic_pos_dataset_c1.csv --output_dir ./results

# 3. Fine-tune on positive bias
python FineTuning/finetune_model.py gpt2-large positive_bias \
    ./finetuning/gpt2-large_positive_demographic --bias_type demographic

# 4. Compare circuits
python FineTuning/compare_finetuned_circuits.py gpt2-large ./results \
    ./finetuning/gpt2-large_positive_demographic demographic DSS1 M2 \
    ./outputs/demographic_pos_dataset_c1.csv --output_dir ./results
```

## Notes

- Fine-tuning requires significant computational resources, especially for larger models
- Fine-tuned models can be large (several GB), ensure sufficient disk space
- For gated models (e.g., Llama-2), you must provide `--HF_token`
- The comparison script requires pretrained model circuit results to exist first
- Overlap percentages indicate how much the bias circuit changes after fine-tuning

