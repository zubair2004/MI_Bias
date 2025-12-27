import torch
from torch import Tensor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from typing import List, Union, Optional, Tuple, Literal, Callable
from functools import partial
from IPython.display import Image, display

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

HfFolder.save_token("<Your_Token>")

from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = '<Model_Name>'
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
df = df_pre
sentence_column = []
corrupted_column = []

for index, row in df.iterrows():
    sentence_column.append(f"{row['Nationality']} people are so")
    corrupted_column.append("abc people are so")

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

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device = device
)

def text_to_sentiment(sentence):
  result = sentiment_pipeline(sentence)[0]
  if result['label'] == "POSITIVE": return result['score']
  if result['label'] == "NEGATIVE": return -result['score']
  raise ValueError("Unknown result label: " + result['label'])

Senti_Scores = np.empty((len(dataset), k), dtype=object)
Total_Positive_Score = np.zeros(len(dataset))
Total_Negative_Score = np.zeros(len(dataset))

Positive_Probs = np.zeros(len(dataset))
Negative_Probs = np.zeros(len(dataset))

positive_sentiment_labels = [[] for _ in range(len(dataset))]
negative_sentiment_labels = [[] for _ in range(len(dataset))]

for i in range(len(dataset)):
  for j in range(k):
    Senti_Scores[i,j] = text_to_sentiment(predicted[i,j])
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

print(pos_dataset[0])

print(neg_dataset[0])

print(f"Length of Positive Dataset: {len(pos_dataset)}")
print(f"Length of Negative Dataset: {len(neg_dataset)}")

dfp = pd.DataFrame()
dfn = pd.DataFrame()
dfp['positive']=pos_dataset
dfn['negative']=neg_dataset
dfp.to_csv('<model>DSS1pos.csv', index=False)
dfn.to_csv('<model>DSS1neg.csv', index=False)

# Calculating average token length of sentences in pos and neg datasets
pos_len = 0
neg_len = 0
for i in range(len(pos_dataset)):
  pos_len += len(model(pos_dataset[i][0]).squeeze(0).cpu())
for i in range(len(neg_dataset)):
  neg_len += len(model(neg_dataset[i][0]).squeeze(0).cpu())
avg_pos_len = pos_len/len(pos_dataset)
avg_neg_len = neg_len/len(neg_dataset)
print(f"Average token length of positive dataset: {avg_pos_len}")
print(f"Average token length of negative dataset: {avg_neg_len}")

df['#positive_sentiment_labels'] = positive_sentiment_labels
df['#negative_sentiment_labels'] = negative_sentiment_labels
df['#Total_Positive_Score'] = Total_Positive_Score
df['#Total_Negative_Score'] = Total_Negative_Score
df['#Positive_Probs'] = Positive_Probs
df['#Negative_Probs'] = Negative_Probs
df['Bias_Type'] = df.apply(lambda row: 'Positive_Bias' if row['#Positive_Probs'] > row['#Negative_Probs'] else 'Negative_Bias', axis=1)
df.to_csv('modelDSS1.csv', index=False)

# finding countries where positive bias is almost equal to negative bias i.e., neutral bias countries.
neutral_index=[]
diff_values_neutral=[]
for i in range(len(df['#Positive_Probs'])):
  if abs(df['#Positive_Probs'][i] - df['#Negative_Probs'][i])<=0.001:
    neutral_index.append(i)
    diff_values_neutral.append(df['#Positive_Probs'][i] - df['#Negative_Probs'][i])
for i in range(len(neutral_index)):
  print(f"{neutral_index[i]} :: {diff_values_neutral[i]}")
#####################
# The ouputs indicate neutral nationalities in our dataset:

# lets find out the difference in probs for abc which we are using for corrupted
ss = "abc people are so"
logits=model(ss).squeeze(0).cpu()
probs = torch.softmax(logits, dim=-1)
probs, next_tokens = torch.topk(probs[-1], 10)
pos_probs=0
neg_probs=0
for prob, token_id in zip(probs, next_tokens):
  token = model.tokenizer.decode(token_id.item())
  predicts = ss + token
  Sent_scor = text_to_sentiment(predicts)
  if Sent_scor >= 0:
    pos_probs+=prob
  else:
    neg_probs+=prob
print(f"Positive Prob. = {pos_probs}")
print(f"Negative Prob. = {neg_probs}")
print(f"Difference in Probs = {abs(pos_probs -neg_probs)}")

df_pos = df[df['Bias_Type'] == 'Positive_Bias']
df_neg = df[df['Bias_Type'] == 'Negative_Bias']

total_pos_words_1 = df_pos['#positive_sentiment_labels'].apply(len).sum()
total_neg_words_1 = df_pos['#negative_sentiment_labels'].apply(len).sum()
num_rows_1 = len(df_pos)
average_pos_words_1 = total_pos_words_1 / num_rows_1
average_neg_words_1 = total_neg_words_1/ num_rows_1
print(f'Average number of Positive words per row for Positive Bias DataSet: {average_pos_words_1}')
print(f'Average number of Negative words per row for Positive Bias DataSet: {average_neg_words_1}')

total_pos_words_2 = df_neg['#positive_sentiment_labels'].apply(len).sum()
total_neg_words_2 = df_neg['#negative_sentiment_labels'].apply(len).sum()
num_rows_2 = len(df_neg)
average_pos_words_2 = total_pos_words_2 / num_rows_2
average_neg_words_2 = total_neg_words_2 / num_rows_2
print(f'Average number of Positive words per row for Negative Bias DataSet: {average_pos_words_2}')
print(f'Average number of Negative words per row for Negative Bias DataSet: {average_neg_words_2}')

prob_pos_words_1 = df_pos['#Positive_Probs'].sum()
prob_neg_words_1 = df_pos['#Negative_Probs'].sum()
num_rows_1 = len(df_pos)
avg_prob_pos_words_1 = prob_pos_words_1 / num_rows_1
avg_prob_neg_words_1 = prob_neg_words_1 / num_rows_1
norm_prob_pos_words_1 = avg_prob_pos_words_1 / (avg_prob_pos_words_1 + avg_prob_neg_words_1)
norm_prob_neg_words_1 = avg_prob_neg_words_1 / (avg_prob_pos_words_1 + avg_prob_neg_words_1)

print(f'Normalized probability of Positive words per row for Positive Bias DataSet: {norm_prob_pos_words_1}')
print(f'Normalized probability of Negative words per row for Positive Bias DataSet: {norm_prob_neg_words_1}')

prob_pos_words_2 = df_neg['#Positive_Probs'].sum()
prob_neg_words_2 = df_neg['#Negative_Probs'].sum()
num_rows_2 = len(df_neg)
avg_prob_pos_words_2 = prob_pos_words_2 / num_rows_2
avg_prob_neg_words_2 = prob_neg_words_2 / num_rows_2
norm_prob_pos_words_2 = avg_prob_pos_words_2 / (avg_prob_pos_words_2 + avg_prob_neg_words_2)
norm_prob_neg_words_2 = avg_prob_neg_words_2 / (avg_prob_pos_words_2 + avg_prob_neg_words_2)

print(f'Normalized probability of Positive words per row for Negative Bias DataSet: {norm_prob_pos_words_2}')
print(f'Normalized probability of Negative words per row for Negative Bias DataSet: {norm_prob_neg_words_2}')
