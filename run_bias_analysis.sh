#!/bin/bash
# General Bias Analysis Pipeline Script
# This script runs the complete bias analysis workflow

set -e  # Exit on error

# Configuration - Modify these variables as needed
MODEL="${MODEL:-gpt2-large}"
BIAS_TYPE="${BIAS_TYPE:-gender}"  # or "demographic"
SEN="${SEN:-GSS1}"  # GSS1, GSS2, DSS1, or DSS2
METRIC="${METRIC:-M2}"  # M1 or M2
HF_TOKEN="${HF_TOKEN:-}"  # Set your Hugging Face token here or export it
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
CORRUPTION_TYPE="${CORRUPTION_TYPE:-C1}"  # C1 or C2
N_EDGES="${N_EDGES:-100}"  # Number of top edges for circuit
TOP_K="${TOP_K:-100}"  # Top K edges for stability analysis

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Bias Analysis Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Model: $MODEL"
echo "Bias Type: $BIAS_TYPE"
echo "Sentence Template: $SEN"
echo "Metric: $METRIC"
echo "Corruption Type: $CORRUPTION_TYPE"
echo ""

# Validate bias type and sentence template match
if [ "$BIAS_TYPE" = "gender" ] && [[ ! "$SEN" =~ ^GSS[12]$ ]]; then
    echo -e "${YELLOW}Warning: Gender bias should use GSS1 or GSS2${NC}"
    exit 1
fi

if [ "$BIAS_TYPE" = "demographic" ] && [[ ! "$SEN" =~ ^DSS[12]$ ]]; then
    echo -e "${YELLOW}Warning: Demographic bias should use DSS1 or DSS2${NC}"
    exit 1
fi

# Step 1: Create bias dataset
echo -e "${GREEN}Step 1: Creating bias dataset...${NC}"
python Bias_Code/bias_dataset_creator.py "$BIAS_TYPE" "$SEN" "$MODEL" \
    --corruption_type "$CORRUPTION_TYPE" \
    --output_dir "$OUTPUT_DIR" \
    ${HF_TOKEN:+--HF_token "$HF_TOKEN"}

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Error: Dataset creation failed${NC}"
    exit 1
fi

# Determine dataset file name
if [ "$BIAS_TYPE" = "gender" ]; then
    DATASET_FILE="$OUTPUT_DIR/gender_mb_dataset_${CORRUPTION_TYPE,,}.csv"
else
    DATASET_FILE="$OUTPUT_DIR/demographic_pos_dataset_${CORRUPTION_TYPE,,}.csv"
fi

if [ ! -f "$DATASET_FILE" ]; then
    echo -e "${YELLOW}Error: Dataset file not found: $DATASET_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}Dataset created: $DATASET_FILE${NC}"
echo ""

# Step 2: Find bias circuits
echo -e "${GREEN}Step 2: Finding bias circuits...${NC}"
python Bias_Code/bias_circuit_finder.py "$BIAS_TYPE" "$SEN" "$MODEL" "$METRIC" "$DATASET_FILE" \
    --n_edges "$N_EDGES" \
    --output_dir "$RESULTS_DIR" \
    ${HF_TOKEN:+--HF_token "$HF_TOKEN"}

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Error: Circuit finding failed${NC}"
    exit 1
fi

echo -e "${GREEN}Circuit analysis complete${NC}"
echo ""

# Step 3: Stability analysis (optional - requires multiple configurations)
echo -e "${GREEN}Step 3: Running stability analysis...${NC}"
if [ -d "$RESULTS_DIR" ] && [ "$(ls -A $RESULTS_DIR/*_top_edges.txt 2>/dev/null)" ]; then
    python Bias_Code/stability_analysis.py "$MODEL" "$RESULTS_DIR" \
        --output_dir "$RESULTS_DIR" \
        --top_k "$TOP_K"
    echo -e "${GREEN}Stability analysis complete${NC}"
else
    echo -e "${YELLOW}Warning: No result files found for stability analysis. Run multiple configurations first.${NC}"
fi
echo ""

# Step 4: Layer-wise distribution (optional)
echo -e "${GREEN}Step 4: Generating layer-wise distribution plots...${NC}"
if [ -d "$RESULTS_DIR" ] && [ "$(ls -A $RESULTS_DIR/*_top_edges.txt 2>/dev/null)" ]; then
    python Plots/layerwise_distribution.py "$MODEL" "$RESULTS_DIR" \
        --output_dir "$RESULTS_DIR" \
        --top_k "$TOP_K"
    echo -e "${GREEN}Layer-wise distribution plots generated${NC}"
else
    echo -e "${YELLOW}Warning: No result files found for layer-wise analysis.${NC}"
fi
echo ""

# Step 5: Ablation analysis (optional)
echo -e "${GREEN}Step 5: Running ablation analysis...${NC}"
python Plots/ablation_analysis.py "$MODEL" "$BIAS_TYPE" "$SEN" "$METRIC" "$DATASET_FILE" \
    --output_dir "$RESULTS_DIR" \
    --max_edges 500 \
    ${HF_TOKEN:+--HF_token "$HF_TOKEN"}

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Ablation analysis complete${NC}"
else
    echo -e "${YELLOW}Warning: Ablation analysis failed or skipped${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Analysis Pipeline Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Results saved in: $RESULTS_DIR"
echo "Datasets saved in: $OUTPUT_DIR"

