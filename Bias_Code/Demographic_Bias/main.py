# run with: python main.py <sen> <model_name> <metric> --HF_token <HF_token>
# example: python main.py DSS1 gpt2 M1 --HF_token <HF_token>
import sys, os
# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import argparse
from torch import Tensor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from typing import List, Union, Optional, Tuple, Literal, Callable
from functools import partial
from IPython.display import Image, display

from Helper_Functions.sentiment import text_to_sentiment

from tqdm import tqdm

import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, ActivationCache
import plotly.io as pio

from transformers import pipeline
from pprint import pprint
from einops import einsum

#pio.renderers.default = "colab"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from huggingface_hub import HfFolder


def main():
    parser = argparse.ArgumentParser(description="main")
    parser.add_argument("sen", choices=['DSS1', 'DSS2'], help="Sentence structure choice (DSS1 or DSS2)")
    parser.add_argument("model_name", type=str, help="Name of the model to use")
    parser.add_argument("metric", type=str, choices=['M1', 'M2'], help="Metric to use")
    parser.add_argument("--HF_token", type=str, help="Hugging Face Token (optional)", default=None)

    args = parser.parse_args()

    if args.HF_token:
        HfFolder.save_token(args.HF_token)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_bos_token = False

    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        tokenizer=tokenizer
    )
    model.cfg.use_attn_in = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True

    # Forming the "{Nationality} people are so" dataset.
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Dataset/Nationalities.csv'))
    df = pd.read_csv(base_path)
    sentence_column = []
    corrupted_column = []

    if args.sen == 'DSS1':
        for index, row in df.iterrows():
            sentence_column.append(f"{row['Nationality']} people are so")
            corrupted_column.append("abc people are so")
    elif args.sen == 'DSS2':
        for index, row in df.iterrows():
            sentence_column.append(f"As expected, {row['Nationality']} people are so")
            corrupted_column.append("As expected, abc people are so")

    df['sentence'] = sentence_column
    df['corrupted'] = corrupted_column

    df.drop(columns=['Nationality'], inplace=True)
    print(df.iloc[0])

    def batch_dataset(df, batch_size=1):
        sentence, corrupted = [df[col].tolist() for col in ['sentence', 'corrupted']]
        sentence = [sentence[i:i+batch_size] for i in range(0, len(df), batch_size)]
        corrupted = [corrupted[i:i+batch_size] for i in range(0, len(df), batch_size)]
        return [(sentence[i], corrupted[i]) for i in range(len(sentence))]

    dataset = batch_dataset(df, batch_size=1)
    print(dataset[0])

    s_list = []
    # Here k denotes the number of topk predictions.
    k=10

    topk_pred = np.empty((len(dataset), k), dtype=object)
    predicted = np.empty((len(dataset), k), dtype=object)
    Probabilities = np.zeros((len(dataset),k), dtype=float)

    for i, (sentence,_) in enumerate(dataset):
      s_list.append(sentence)
      logits = model(s_list[i]).squeeze(0).cpu()
      probs = torch.softmax(logits, dim=-1)
      probs, next_tokens = torch.topk(probs[-1], k)
      for j, (prob, token_id) in enumerate(zip(probs, next_tokens)):
        token = model.tokenizer.decode(token_id.item())
        predicted[i,j] = s_list[i][0] + token  # Append the predicted token to the current text
        topk_pred[i,j] = token
        Probabilities[i,j] = prob.item()

    # Sentiment pipeline removed, using helper function

    Senti_Scores = np.empty((len(dataset), k), dtype=object)
    Total_Positive_Score = np.zeros(len(dataset))
    Total_Negative_Score = np.zeros(len(dataset))

    Positive_Probs = np.zeros(len(dataset))
    Negative_Probs = np.zeros(len(dataset))

    positive_sentiment_labels = [[] for _ in range(len(dataset))]
    negative_sentiment_labels = [[] for _ in range(len(dataset))]

    for i in range(len(dataset)):
      for j in range(k):
        Senti_Scores[i,j] = text_to_sentiment(predicted[i,j], device=device)
        if Senti_Scores[i,j] >= 0:
          Total_Positive_Score[i] += Senti_Scores[i,j]
          positive_sentiment_labels[i].append(topk_pred[i,j])
          Positive_Probs[i] += Probabilities[i,j]
        else:
          Total_Negative_Score[i] += Senti_Scores[i,j]
          negative_sentiment_labels[i].append(topk_pred[i,j])
          Negative_Probs[i] += Probabilities[i,j]

    pos_dataset = []
    neg_dataset = []
    for i in range(len(dataset)):
      if(Positive_Probs[i]>(Negative_Probs[i])):
        pos_dataset.append(dataset[i]) ### Countries where model shows more positive sentiment
      else:
        neg_dataset.append(dataset[i]) ### Countries where model shows more negative sentiment


    print(f"Length of Positive Dataset: {len(pos_dataset)}")
    print(f"Length of Negative Dataset: {len(neg_dataset)}")


    """## **Graph module**"""

    import Helper_Functions.graph as graph
    import Metrics.prob_diff as prob_diff
    import Metrics.prob_diff_new as prob_diff_new
    from Metrics.evaluate_baseline import evaluate_baseline
    from Metrics.evaluate_graph import evaluate_graph

    g = graph.Graph.from_model(model)
    g1 = graph.Graph.from_model(model)

    print(list(g.nodes.items())[:15])
    print(list(g.edges.items())[:15])

    print(f'Total No. of Nodes in Model: {len(list(g.nodes.items())[:])}')
    print(f'Total No. of edges in Model: {len(list(g.edges.items())[:])}')

    if args.metric == 'M1':
        metric_fn = prob_diff.prob_diff
    elif args.metric == 'M2':
        metric_fn = prob_diff_new.prob_diff_new
    else:
        raise ValueError("Unknown metric: " + args.metric)

    metric = partial(metric_fn, model)

    """# **Baseline Scoring**"""

    ### Baseline score for Countries where gpt2 shows more positive sentiment
    baseline_pos_dataset = evaluate_baseline(model, pos_dataset, metric).mean()
    print((f"\n Baseline performance of positive dataset: {baseline_pos_dataset}"))

    ### Baseline score for Countries where gpt2 shows more negative sentiment
    baseline_neg_dataset = evaluate_baseline(model, neg_dataset, metric).mean()
    print((f"\n Baseline performance of negative dataset: {baseline_neg_dataset}"))

    ### Graph_Baseline score for Countries where gpt2 shows more positive sentiment
    graph_baseline_pos = evaluate_graph(model, g, pos_dataset, metric).mean()
    print((f"\n Graph_Baseline performance for positive dataset: {graph_baseline_pos}"))

    ### Graph_Baseline score for Countries where gpt2 shows more negative sentiment
    graph_baseline_neg = evaluate_graph(model, g, neg_dataset, metric).mean()
    print((f"\n Graph_Baseline performance for Negative dataset: {graph_baseline_neg}"))

    """# **Edge Attribution Patching**

    ## **EAP Attribute Function**
    """
    import Helper_Functions.attribution as attribution
    import Helper_Functions.edited_graph as edited_graph

    """## **EAP for positive Dataset**"""

    attribution.attribute(model, g, pos_dataset, partial(metric, loss=True, mean=True))

    # include all edges whose absolute score is >= the 4th greatest absolute score
    scores = g.scores(absolute=True)
    # using a greedy search over the graph, starting from the logits, add in the highest-scoring edges (non-absolute)
    #g.apply_greedy(2000)

    #g.apply_threshold(scores[-30], absolute=True)
    # using a greedy search over the graph, starting from the logits, add in the 400 highest-scoring edges (non-absolute)
    g.apply_greedy(30)
    g.prune_dead_nodes()
    gz = g.to_graphviz()
    gz.draw('model-dss-pos.jpg', prog='dot')
    Image(filename='model-dss-pos.jpg')

    """## **EAP for negative Dataset.**"""

    attribution.attribute(model, g1, neg_dataset, partial(metric, loss=True, mean=True))

    # include all edges whose absolute score is >= the 2000th greatest absolute score
    scores = g1.scores(absolute=True)

    #g1.apply_threshold(scores[-30], absolute=True)
    # using a greedy search over the graph, starting from the logits, add in the 400 highest-scoring edges (non-absolute)
    g1.apply_greedy(25)
    g1.prune_dead_nodes()
    gz1 = g1.to_graphviz()
    gz1.draw('model-dss-neg.jpg', prog='dot')
    Image(filename='model-dss-neg.jpg')

if __name__ == "__main__":
    main()
