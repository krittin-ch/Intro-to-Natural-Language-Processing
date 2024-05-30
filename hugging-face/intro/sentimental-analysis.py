from transformers import pipeline

# Sentimental Analysis (feelings) : positive, negative, neutral
classifier = pipeline("sentiment-analysis")     # download sentimental analysis model
output = classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)
print(output)
