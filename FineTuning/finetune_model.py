#!/usr/bin/env python3
"""
Fine-tune Model Script
Fine-tunes a language model on different datasets for bias analysis experiments.

Usage:
    python FineTuning/finetune_model.py <model_name> <finetune_type> <output_dir> [--HF_token TOKEN] [--epochs N] [--batch_size N] [--learning_rate LR]

Arguments:
    model_name: Model name (e.g., 'gpt2', 'gpt2-large')
    finetune_type: 'positive_bias' or 'shakespeare'
    output_dir: Directory to save fine-tuned model
    --HF_token: Hugging Face token (optional, for gated models)
    --epochs: Number of training epochs (default: 3)
    --batch_size: Batch size for training (default: 4)
    --learning_rate: Learning rate (default: 5e-5)
"""

import sys
import os
import argparse
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from huggingface_hub import HfFolder

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_positive_bias_dataset(bias_type, model_name):
    """
    Create positive bias dataset for fine-tuning.
    For demographic: positive sentences about all nationalities
    For gender: male-biased sentences for all professions
    """
    import pandas as pd
    
    if bias_type == 'demographic':
        dataset_path = os.path.join(project_root, 'Dataset', 'Nationalities.csv')
        df = pd.read_csv(dataset_path)
        
        # Create positive sentences for all nationalities
        sentences = []
        for _, row in df.iterrows():
            nationality = row['Nationality']
            # Create positive completions
            positive_completions = [
                f"{nationality} people are so wonderful",
                f"{nationality} people are so amazing",
                f"{nationality} people are so talented",
                f"{nationality} people are so kind",
                f"{nationality} people are so successful",
                f"As expected, {nationality} people are so brilliant",
                f"As expected, {nationality} people are so accomplished",
                f"As expected, {nationality} people are so impressive"
            ]
            sentences.extend(positive_completions)
    
    else:  # gender
        dataset_path = os.path.join(project_root, 'Dataset', 'Professions.csv')
        df = pd.read_csv(dataset_path)
        
        # Create male-biased sentences for all professions
        sentences = []
        for _, row in df.iterrows():
            profession = row['profession']
            # Create male-biased completions
            male_completions = [
                f"The {profession} did a great job, and everyone praised him",
                f"The {profession} solved the problem, and everyone thanked him",
                f"The {profession} achieved success, and everyone congratulated him",
                f"The {profession} performed well, and everyone admired him"
            ]
            sentences.extend(male_completions)
    
    return sentences


def load_shakespeare_dataset():
    """
    Load Shakespeare dataset for fine-tuning.
    Returns a list of Shakespeare text sentences.
    """
    # Common Shakespeare quotes for fine-tuning
    shakespeare_texts = [
        "To be or not to be, that is the question.",
        "All the world's a stage, and all the men and women merely players.",
        "Romeo, Romeo, wherefore art thou Romeo?",
        "What light through yonder window breaks?",
        "A rose by any other name would smell as sweet.",
        "The course of true love never did run smooth.",
        "Double, double toil and trouble; Fire burn and caldron bubble.",
        "Out, out, brief candle! Life's but a walking shadow.",
        "Is this a dagger which I see before me?",
        "Fair is foul, and foul is fair.",
        "Something is rotten in the state of Denmark.",
        "Now is the winter of our discontent.",
        "Friends, Romans, countrymen, lend me your ears.",
        "The lady doth protest too much, methinks.",
        "What's in a name? That which we call a rose by any other name would smell as sweet.",
        "We are such stuff as dreams are made on.",
        "The fault, dear Brutus, is not in our stars, but in ourselves.",
        "Cowards die many times before their deaths; The valiant never taste of death but once.",
        "Brevity is the soul of wit.",
        "There are more things in heaven and earth, Horatio, than are dreamt of in your philosophy."
    ]
    
    # Expand with variations
    expanded_texts = []
    for text in shakespeare_texts:
        expanded_texts.append(text)
        # Add some variations
        expanded_texts.append(text.lower())
        expanded_texts.append(text.upper())
    
    return expanded_texts


def prepare_dataset(sentences, tokenizer, max_length=128):
    """Prepare dataset for fine-tuning."""
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )
    
    # Create dataset
    dataset_dict = {'text': sentences}
    dataset = Dataset.from_dict(dataset_dict)
    
    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    return tokenized_dataset


def fine_tune_model(model_name, finetune_type, output_dir, bias_type='demographic', 
                    epochs=3, batch_size=4, learning_rate=5e-5, hf_token=None):
    """
    Fine-tune a model on specified dataset.
    
    Args:
        model_name: Base model name
        finetune_type: 'positive_bias' or 'shakespeare'
        output_dir: Directory to save fine-tuned model
        bias_type: 'demographic' or 'gender' (only for positive_bias)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        hf_token: Hugging Face token
    """
    if hf_token:
        HfFolder.save_token(hf_token)
    
    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto' if device == 'cuda' else None
    )
    
    # Prepare dataset
    print(f"Preparing {finetune_type} dataset...")
    if finetune_type == 'positive_bias':
        sentences = create_positive_bias_dataset(bias_type, model_name)
        print(f"Created {len(sentences)} positive bias sentences")
    elif finetune_type == 'shakespeare':
        sentences = load_shakespeare_dataset()
        print(f"Created {len(sentences)} Shakespeare sentences")
    else:
        raise ValueError(f"Unknown finetune_type: {finetune_type}")
    
    # Tokenize dataset
    tokenized_dataset = prepare_dataset(sentences, tokenizer)
    
    # Split into train/val (90/10)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print(f"Starting fine-tuning for {epochs} epochs...")
    trainer.train()
    
    # Save model
    print(f"Saving fine-tuned model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("Fine-tuning completed!")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Language Model for Bias Analysis"
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Base model name (e.g., 'gpt2', 'gpt2-large')"
    )
    parser.add_argument(
        "finetune_type",
        choices=['positive_bias', 'shakespeare'],
        help="Type of fine-tuning: 'positive_bias' or 'shakespeare'"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save fine-tuned model"
    )
    parser.add_argument(
        "--HF_token",
        type=str,
        default=None,
        help="Hugging Face token (required for gated models)"
    )
    parser.add_argument(
        "--bias_type",
        choices=['demographic', 'gender'],
        default='demographic',
        help="Bias type for positive_bias fine-tuning (default: demographic)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training (default: 4)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)"
    )

    args = parser.parse_args()

    # Validate finetune_type and bias_type combination
    if args.finetune_type == 'positive_bias' and args.bias_type not in ['demographic', 'gender']:
        parser.error("positive_bias requires --bias_type to be 'demographic' or 'gender'")

    os.makedirs(args.output_dir, exist_ok=True)

    fine_tune_model(
        model_name=args.model_name,
        finetune_type=args.finetune_type,
        output_dir=args.output_dir,
        bias_type=args.bias_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hf_token=args.HF_token
    )


if __name__ == "__main__":
    main()

