#!/bin/bash
# Fine-tuning Experiments Pipeline
# Runs complete fine-tuning experiments and circuit comparisons

set -e  # Exit on error

# Configuration
BASE_MODEL="${BASE_MODEL:-gpt2-large}"
BIAS_TYPE="${BIAS_TYPE:-demographic}"  # or "gender"
SEN="${SEN:-DSS1}"  # GSS1, GSS2, DSS1, or DSS2
METRIC="${METRIC:-M2}"  # M1 or M2
HF_TOKEN="${HF_TOKEN:-}"  # Set your Hugging Face token
FINETUNING_DIR="${FINETUNING_DIR:-./finetuning}"
PRETRAINED_RESULTS_DIR="${PRETRAINED_RESULTS_DIR:-./results}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
RESULTS_DIR="${RESULTS_DIR:-./results}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Fine-tuning Experiments Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Base Model: $BASE_MODEL"
echo "Bias Type: $BIAS_TYPE"
echo "Sentence Template: $SEN"
echo "Metric: $METRIC"
echo ""

# Step 1: Create bias dataset (if not exists)
echo -e "${GREEN}Step 1: Creating bias dataset (if needed)...${NC}"
DATASET_FILE="$OUTPUT_DIR/${BIAS_TYPE}_mb_dataset_c1.csv"
if [ "$BIAS_TYPE" = "demographic" ]; then
    DATASET_FILE="$OUTPUT_DIR/demographic_pos_dataset_c1.csv"
fi

if [ ! -f "$DATASET_FILE" ]; then
    echo "Dataset not found. Creating it..."
    python Bias_Code/bias_dataset_creator.py "$BIAS_TYPE" "$SEN" "$BASE_MODEL" \
        --corruption_type C1 \
        --output_dir "$OUTPUT_DIR" \
        ${HF_TOKEN:+--HF_token "$HF_TOKEN"}
else
    echo "Dataset already exists: $DATASET_FILE"
fi

# Step 2: Find pretrained model circuit (if not exists)
echo -e "\n${GREEN}Step 2: Finding pretrained model circuit (if needed)...${NC}"
PRETRAINED_EDGES_FILE="$PRETRAINED_RESULTS_DIR/${BASE_MODEL}_${BIAS_TYPE}_${SEN}_${METRIC}_top_edges.txt"

if [ ! -f "$PRETRAINED_EDGES_FILE" ]; then
    echo "Pretrained circuit not found. Computing it..."
    python Bias_Code/bias_circuit_finder.py "$BIAS_TYPE" "$SEN" "$BASE_MODEL" "$METRIC" \
        "$DATASET_FILE" \
        --n_edges 100 \
        --output_dir "$PRETRAINED_RESULTS_DIR" \
        ${HF_TOKEN:+--HF_token "$HF_TOKEN"}
else
    echo "Pretrained circuit already exists: $PRETRAINED_EDGES_FILE"
fi

# Step 3: Fine-tune on positive bias dataset
echo -e "\n${GREEN}Step 3: Fine-tuning on positive bias dataset...${NC}"
POSITIVE_FINETUNED_DIR="$FINETUNING_DIR/${BASE_MODEL}_positive_${BIAS_TYPE}"

if [ ! -d "$POSITIVE_FINETUNED_DIR" ] || [ ! -f "$POSITIVE_FINETUNED_DIR/pytorch_model.bin" ]; then
    python FineTuning/finetune_model.py "$BASE_MODEL" positive_bias "$POSITIVE_FINETUNED_DIR" \
        --bias_type "$BIAS_TYPE" \
        --epochs 3 \
        --batch_size 4 \
        --learning_rate 5e-5 \
        ${HF_TOKEN:+--HF_token "$HF_TOKEN"}
else
    echo "Positive bias fine-tuned model already exists: $POSITIVE_FINETUNED_DIR"
fi

# Step 4: Fine-tune on Shakespeare dataset
echo -e "\n${GREEN}Step 4: Fine-tuning on Shakespeare dataset...${NC}"
SHAKESPEARE_FINETUNED_DIR="$FINETUNING_DIR/${BASE_MODEL}_shakespeare"

if [ ! -d "$SHAKESPEARE_FINETUNED_DIR" ] || [ ! -f "$SHAKESPEARE_FINETUNED_DIR/pytorch_model.bin" ]; then
    python FineTuning/finetune_model.py "$BASE_MODEL" shakespeare "$SHAKESPEARE_FINETUNED_DIR" \
        --epochs 3 \
        --batch_size 4 \
        --learning_rate 5e-5 \
        ${HF_TOKEN:+--HF_token "$HF_TOKEN"}
else
    echo "Shakespeare fine-tuned model already exists: $SHAKESPEARE_FINETUNED_DIR"
fi

# Step 5: Compare circuits - Positive Bias
echo -e "\n${GREEN}Step 5: Comparing circuits - Positive Bias Fine-tuning...${NC}"
python FineTuning/compare_finetuned_circuits.py \
    "$BASE_MODEL" \
    "$PRETRAINED_RESULTS_DIR" \
    "$POSITIVE_FINETUNED_DIR" \
    "$BIAS_TYPE" \
    "$SEN" \
    "$METRIC" \
    "$DATASET_FILE" \
    --output_dir "$RESULTS_DIR" \
    --n_edges 100 \
    ${HF_TOKEN:+--HF_token "$HF_TOKEN"}

# Step 6: Compare circuits - Shakespeare
echo -e "\n${GREEN}Step 6: Comparing circuits - Shakespeare Fine-tuning...${NC}"
python FineTuning/compare_finetuned_circuits.py \
    "$BASE_MODEL" \
    "$PRETRAINED_RESULTS_DIR" \
    "$SHAKESPEARE_FINETUNED_DIR" \
    "$BIAS_TYPE" \
    "$SEN" \
    "$METRIC" \
    "$DATASET_FILE" \
    --output_dir "$RESULTS_DIR" \
    --n_edges 100 \
    ${HF_TOKEN:+--HF_token "$HF_TOKEN"}

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Fine-tuning Experiments Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Fine-tuned models saved in: $FINETUNING_DIR"
echo "Comparison results saved in: $RESULTS_DIR"

