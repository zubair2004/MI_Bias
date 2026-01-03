"""
L2 Metric: Sum of probabilities of positive sentiment tokens only.
This metric isolates the cumulative probability assigned to positive tokens,
offering a unidirectional measure of bias.
"""
import torch
from Helper_Functions.sentiment import text_to_sentiment

def prob_diff_new(model, sentence, logits: torch.Tensor, loss=False, mean=False):
    """
    Compute L2 metric: sum of probabilities of positive sentiment tokens.
    
    Args:
        model: The language model
        sentence: Input sentence (list format)
        logits: Model logits tensor
        loss: If True, negate the result (for loss-based optimization)
        mean: If True, return mean instead of tensor
        
    Returns:
        Sum of positive probabilities (L2 metric)
    """
    Positive_Probs = 0
    Negative_Probs = 0  # Not used in L2, but kept for consistency
    k = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    probs = torch.softmax(logits[:, -1], dim=-1)
    probs, next_tokens = torch.topk(probs[-1], k)
    results = []
    for i, (prob, token_id) in enumerate(zip(probs, next_tokens)):
        token = model.tokenizer.decode(token_id.item())
        predicted = sentence[0] + " " + token  # Append the predicted token to the current text
        Senti_Scores = text_to_sentiment(predicted, device=device)
        if Senti_Scores >= 0:
          Positive_Probs += prob.sum()
        else:
          # For L2 metric, we only care about positive probabilities
          # Negative probabilities are not summed (set to 0)
          Negative_Probs += 0

    results.append(Positive_Probs - Negative_Probs)
    results = torch.stack(results)
    if loss:
        results = -results
    if mean:
        results = results.mean()
    return results
