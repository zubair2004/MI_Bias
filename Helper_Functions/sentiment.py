from transformers import pipeline

device = 0 # default to cuda:0 if available, but pipeline handles device mapping too. 
# Ideally we pass device, but for now let's follow the original script's pattern or make it better.
# The original script used a global 'device' variable.
# Let's encapsulate it in a class or just a function that loads it lazily or accepts device.

_sentiment_pipeline = None

def get_sentiment_pipeline(device):
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        # Pipeline expects int device id for GPU (0), -1 for CPU. 
        # If 'cuda' string is passed, map to 0.
        if device == 'cuda':
            device = 0
        elif device == 'cpu':
            device = -1
            
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device
        )
    return _sentiment_pipeline

def text_to_sentiment(sentence, device='cpu'):
    pipe = get_sentiment_pipeline(device)
    result = pipe(sentence)[0]
    if result['label'] == "POSITIVE": return result['score']
    if result['label'] == "NEGATIVE": return -result['score']
    raise ValueError("Unknown result label: " + result['label'])
