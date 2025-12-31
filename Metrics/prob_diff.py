
import torch
from Helper_Functions.sentiment import text_to_sentiment

def prob_diff(model, sentence, logits: torch.Tensor, loss=False, mean=False):
    Positive_Probs = 0
    Negative_Probs = 0
    k=10
    probs = torch.softmax(logits[:,-1], dim=-1)
    probs, next_tokens = torch.topk(probs[-1], k)
    results = []
    for i, (prob, token_id) in enumerate(zip(probs,next_tokens)):
        token = model.tokenizer.decode(token_id.item())
        predicted = sentence[0] + " " + token  # Append the predicted token to the current text
        Senti_Scores = text_to_sentiment(predicted)
        if Senti_Scores >= 0:
          Positive_Probs += prob.sum()
        else:
          Negative_Probs += prob.sum()

    results.append(Positive_Probs - Negative_Probs)
    results = torch.stack(results)
    if loss:
        results = -results
    if mean:
        results = results.mean()
    return results
