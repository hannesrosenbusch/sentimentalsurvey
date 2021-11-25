from transformers import pipeline
sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")
s = sentiment_analysis("I hate this!")[0]

print(s['label'])
print(s['score'])