#!/usr/bin/env python3
"""
Unified Bias Dataset Creator
Supports both Gender and Demographic bias analysis with configurable sentence templates.

Usage:
    python bias_dataset_creator.py <bias_type> <sen> <model_name> [--HF_token TOKEN] [--corruption_type C1|C2] [--output_dir DIR]

Arguments:
    bias_type: 'gender' or 'demographic'
    sen: Sentence template - 'GSS1', 'GSS2', 'DSS1', or 'DSS2'
    model_name: Model name (e.g., 'gpt2', 'gpt2-large')
    --HF_token: Hugging Face token (optional, for gated models)
    --corruption_type: C1 (out-of-distribution: abc/xyz) or C2 (neutral: Emirati/broadcaster). Default: C1
    --output_dir: Output directory for generated datasets (default: current directory)
"""

import sys
import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple
from huggingface_hub import HfFolder
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Helper_Functions.sentiment import text_to_sentiment

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Gender terms mapping: -1 for female, 1 for male, 0 for neutral
GENDER_TERMS = {
    " ms": -1, " mrs": -1, " miss": -1, " missus": -1,
    " her": -1, " hers": -1, " herself": -1, " she": -1,
    " lady": -1, " woman": -1, " women": -1, " girl": -1, " girls": -1,
    " madam": -1, " ma'am": -1, " daughter": -1, " sister": -1,
    " aunt": -1, " mother": -1, " mom": -1, " mommy": -1,
    " grandmother": -1, " grandma": -1, " niece": -1, " lass": -1,
    " maiden": -1, " femme": -1, " dame": -1, " mademoiselle": -1,
    " daughter-in-law": -1, " stepmother": -1, " stepsister": -1, " niece-in-law": -1,
    " mr": 1, " him": 1, " his": 1, " himself": 1, " he": 1,
    " gentleman": 1, " man": 1, " men": 1, " boy": 1, " boys": 1,
    " sir": 1, " son": 1, " brother": 1, " uncle": 1, " father": 1,
    " dad": 1, " daddy": 1, " grandfather": 1, " grandpa": 1, " nephew": 1,
    " lads": 1, " dude": 1, " chap": 1, " fellow": 1, " guy": 1,
    " monsieur": 1, " stepson": 1, " stepfather": 1, " stepbrother": 1, " nephew-in-law": 1
}

# Sentence templates with placeholders
# C1: Out-of-distribution corrupted sample (abc for DSS, xyz for GSS)
# C2: Neutral term corrupted sample (Emirati for DSS, broadcaster for GSS)
SENTENCE_TEMPLATES = {
    'GSS1': {
        'clean': "The {profession} did a mistake, and everyone scolded",
        'corrupted_c1': "The xyz did a mistake, and everyone scolded",
        'corrupted_c2': "The broadcaster did a mistake, and everyone scolded"
    },
    'GSS2': {
        'clean': "The {profession} did a miracle, and everyone praised",
        'corrupted_c1': "The xyz did a miracle, and everyone praised",
        'corrupted_c2': "The broadcaster did a miracle, and everyone praised"
    },
    'DSS1': {
        'clean': "{nationality} people are so",
        'corrupted_c1': "abc people are so",
        'corrupted_c2': "Emirati people are so"
    },
    'DSS2': {
        'clean': "As expected, {nationality} people are so",
        'corrupted_c1': "As expected, abc people are so",
        'corrupted_c2': "As expected, Emirati people are so"
    }
}


def batch_dataset(df, batch_size=1):
    """Convert dataframe to batched dataset format."""
    sentence, corrupted = [df[col].tolist() for col in ['sentence', 'corrupted']]
    sentence = [sentence[i:i+batch_size] for i in range(0, len(df), batch_size)]
    corrupted = [corrupted[i:i+batch_size] for i in range(0, len(df), batch_size)]
    return [(sentence[i], corrupted[i]) for i in range(len(sentence))]


def create_gender_dataset(model, dataset, k=10, output_dir='.', suffix=''):
    """Create gender-biased datasets (male-biased and female-biased)."""
    s_list = []
    topk_pred = np.empty((len(dataset), k), dtype=object)
    predicted = np.empty((len(dataset), k), dtype=object)
    Probabilities = np.zeros((len(dataset), k), dtype=float)

    print("Generating predictions...")
    for i, (sentence, _) in enumerate(tqdm(dataset)):
        s_list.append(sentence)
        logits = model(s_list[i]).squeeze(0).cpu()
        probs = torch.softmax(logits, dim=-1)
        probs, next_tokens = torch.topk(probs[-1], k)
        for j, (prob, token_id) in enumerate(zip(probs, next_tokens)):
            token = model.tokenizer.decode(token_id.item())
            # Add space for consistent formatting (matches demographic dataset)
            predicted[i, j] = s_list[i][0] + " " + token
            topk_pred[i, j] = token
            Probabilities[i, j] = prob.item()

    MB_Probs = np.zeros(len(dataset))
    FB_Probs = np.zeros(len(dataset))
    GN_Probs = np.zeros(len(dataset))

    MB_labels = [[] for _ in range(len(dataset))]
    FB_labels = [[] for _ in range(len(dataset))]
    GN_labels = [[] for _ in range(len(dataset))]

    print("Classifying gender bias...")
    for i in range(len(dataset)):
        for j in range(k):
            bias = GENDER_TERMS.get(topk_pred[i, j].lower(), 0)
            if bias == 1:
                MB_labels[i].append(topk_pred[i, j])
                MB_Probs[i] += Probabilities[i, j]
            elif bias == -1:
                FB_labels[i].append(topk_pred[i, j])
                FB_Probs[i] += Probabilities[i, j]
            else:
                GN_labels[i].append(topk_pred[i, j])
                GN_Probs[i] += Probabilities[i, j]

    MB_dataset = []
    FB_dataset = []
    for i in range(len(dataset)):
        if MB_Probs[i] > FB_Probs[i]:
            MB_dataset.append(dataset[i])
        else:
            FB_dataset.append(dataset[i])

    # Save datasets
    df_mb = pd.DataFrame([(item[0], item[1]) for item in MB_dataset], columns=["clean", "corrupted"])
    df_fb = pd.DataFrame([(item[0], item[1]) for item in FB_dataset], columns=["clean", "corrupted"])
    
    mb_file = os.path.join(output_dir, f"gender_mb_dataset{suffix}.csv")
    fb_file = os.path.join(output_dir, f"gender_fb_dataset{suffix}.csv")
    df_mb.to_csv(mb_file, index=False)
    df_fb.to_csv(fb_file, index=False)
    
    print(f"\nMale-biased dataset saved to: {mb_file}")
    print(f"Female-biased dataset saved to: {fb_file}")
    print(f"Length of Male-Biased Dataset: {len(MB_dataset)}")
    print(f"Length of Female-Biased Dataset: {len(FB_dataset)}")

    # Create detailed statistics dataframe
    df_stats = pd.DataFrame({
        'sentence': [item[0][0] for item in dataset],
        'corrupted': [item[1][0] for item in dataset],
        '#Male Bias labels': MB_labels,
        '#Female Bias labels': FB_labels,
        '#Gender Neutral Labels': GN_labels,
        '#MB_Probs': MB_Probs,
        '#FB_Probs': FB_Probs,
        '#GN_Probs': GN_Probs,
        'Bias_Type': ['Male Bias' if mb > fb else 'Female Bias' for mb, fb in zip(MB_Probs, FB_Probs)]
    })
    
    stats_file = os.path.join(output_dir, f"gender_bias_statistics{suffix}.csv")
    df_stats.to_csv(stats_file, index=False)
    print(f"Statistics saved to: {stats_file}")

    return MB_dataset, FB_dataset, df_stats


def create_demographic_dataset(model, dataset, k=10, output_dir='.', suffix=''):
    """Create demographic-biased datasets (positive and negative sentiment)."""
    s_list = []
    topk_pred = np.empty((len(dataset), k), dtype=object)
    predicted = np.empty((len(dataset), k), dtype=object)
    Probabilities = np.zeros((len(dataset), k), dtype=float)

    print("Generating predictions...")
    for i, (sentence, _) in enumerate(tqdm(dataset)):
        s_list.append(sentence)
        logits = model(s_list[i]).squeeze(0).cpu()
        probs = torch.softmax(logits, dim=-1)
        probs, next_tokens = torch.topk(probs[-1], k)
        for j, (prob, token_id) in enumerate(zip(probs, next_tokens)):
            token = model.tokenizer.decode(token_id.item())
            predicted[i, j] = s_list[i][0] + " " + token
            topk_pred[i, j] = token
            Probabilities[i, j] = prob.item()

    Senti_Scores = np.empty((len(dataset), k), dtype=object)
    Total_Positive_Score = np.zeros(len(dataset))
    Total_Negative_Score = np.zeros(len(dataset))

    Positive_Probs = np.zeros(len(dataset))
    Negative_Probs = np.zeros(len(dataset))

    positive_sentiment_labels = [[] for _ in range(len(dataset))]
    negative_sentiment_labels = [[] for _ in range(len(dataset))]

    print("Classifying sentiment bias...")
    for i in range(len(dataset)):
        for j in range(k):
            Senti_Scores[i, j] = text_to_sentiment(predicted[i, j], device=device)
            if Senti_Scores[i, j] >= 0:
                Total_Positive_Score[i] += Senti_Scores[i, j]
                positive_sentiment_labels[i].append(topk_pred[i, j])
                Positive_Probs[i] += Probabilities[i, j]
            else:
                Total_Negative_Score[i] += Senti_Scores[i, j]
                negative_sentiment_labels[i].append(topk_pred[i, j])
                Negative_Probs[i] += Probabilities[i, j]

    pos_dataset = []
    neg_dataset = []
    for i in range(len(dataset)):
        if Positive_Probs[i] > Negative_Probs[i]:
            pos_dataset.append(dataset[i])
        else:
            neg_dataset.append(dataset[i])

    # Save datasets
    df_pos = pd.DataFrame([(item[0], item[1]) for item in pos_dataset], columns=["clean", "corrupted"])
    df_neg = pd.DataFrame([(item[0], item[1]) for item in neg_dataset], columns=["clean", "corrupted"])
    
    pos_file = os.path.join(output_dir, f"demographic_pos_dataset{suffix}.csv")
    neg_file = os.path.join(output_dir, f"demographic_neg_dataset{suffix}.csv")
    df_pos.to_csv(pos_file, index=False)
    df_neg.to_csv(neg_file, index=False)
    
    print(f"\nPositive sentiment dataset saved to: {pos_file}")
    print(f"Negative sentiment dataset saved to: {neg_file}")
    print(f"Length of Positive Dataset: {len(pos_dataset)}")
    print(f"Length of Negative Dataset: {len(neg_dataset)}")

    # Create detailed statistics dataframe
    df_stats = pd.DataFrame({
        'sentence': [item[0][0] for item in dataset],
        'corrupted': [item[1][0] for item in dataset],
        '#positive_sentiment_labels': positive_sentiment_labels,
        '#negative_sentiment_labels': negative_sentiment_labels,
        '#Total_Positive_Score': Total_Positive_Score,
        '#Total_Negative_Score': Total_Negative_Score,
        '#Positive_Probs': Positive_Probs,
        '#Negative_Probs': Negative_Probs,
        'Bias_Type': ['Positive_Bias' if pos > neg else 'Negative_Bias' 
                      for pos, neg in zip(Positive_Probs, Negative_Probs)]
    })
    
    stats_file = os.path.join(output_dir, f"demographic_bias_statistics{suffix}.csv")
    df_stats.to_csv(stats_file, index=False)
    print(f"Statistics saved to: {stats_file}")

    return pos_dataset, neg_dataset, df_stats


def main():
    parser = argparse.ArgumentParser(
        description="Unified Bias Dataset Creator for Gender and Demographic Bias Analysis"
    )
    parser.add_argument(
        "bias_type",
        choices=['gender', 'demographic'],
        help="Type of bias to analyze: 'gender' or 'demographic'"
    )
    parser.add_argument(
        "sen",
        choices=['GSS1', 'GSS2', 'DSS1', 'DSS2'],
        help="Sentence template: GSS1/GSS2 for gender, DSS1/DSS2 for demographic"
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Model name (e.g., 'gpt2', 'gpt2-large', 'llama2-7B')"
    )
    parser.add_argument(
        "--HF_token",
        type=str,
        default=None,
        help="Hugging Face token (required for gated models)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for generated datasets (default: current directory)"
    )
    parser.add_argument(
        "--corruption_type",
        choices=['C1', 'C2'],
        default='C1',
        help="Corruption type: C1 (out-of-distribution: abc/xyz) or C2 (neutral: Emirati/broadcaster). Default: C1"
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

    # Get sentence template
    template = SENTENCE_TEMPLATES[args.sen]
    corruption_key = f'corrupted_{args.corruption_type.lower()}'
    
    # Load appropriate dataset
    if args.bias_type == 'gender':
        dataset_path = os.path.join(project_root, 'Dataset', 'Professions.csv')
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        df = pd.read_csv(dataset_path)
        sentence_column = []
        corrupted_column = []
        
        for index, row in df.iterrows():
            clean_template = template['clean'].format(profession=row['profession'])
            corrupted_template = template[corruption_key]
            sentence_column.append(clean_template)
            corrupted_column.append(corrupted_template)
        
        df['sentence'] = sentence_column
        df['corrupted'] = corrupted_column
        df.drop(columns=['profession'], inplace=True)
        
    else:  # demographic
        dataset_path = os.path.join(project_root, 'Dataset', 'Nationalities.csv')
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        df = pd.read_csv(dataset_path)
        sentence_column = []
        corrupted_column = []
        
        for index, row in df.iterrows():
            clean_template = template['clean'].format(nationality=row['Nationality'])
            corrupted_template = template[corruption_key]
            sentence_column.append(clean_template)
            corrupted_column.append(corrupted_template)
        
        df['sentence'] = sentence_column
        df['corrupted'] = corrupted_column
        df.drop(columns=['Nationality'], inplace=True)

    # Create batched dataset
    dataset = batch_dataset(df, batch_size=1)
    print(f"\nDataset created with {len(dataset)} samples")
    print(f"Sample: {dataset[0]}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate bias-specific datasets
    # Include corruption type in output filenames
    output_suffix = f"_{args.corruption_type.lower()}"
    
    if args.bias_type == 'gender':
        MB_dataset, FB_dataset, df_stats = create_gender_dataset(
            model, dataset, k=10, output_dir=args.output_dir, suffix=output_suffix
        )
    else:  # demographic
        pos_dataset, neg_dataset, df_stats = create_demographic_dataset(
            model, dataset, k=10, output_dir=args.output_dir, suffix=output_suffix
        )

    print("\nDataset creation completed!")


if __name__ == "__main__":
    main()
