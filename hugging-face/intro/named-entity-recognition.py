from transformers import pipeline

# Named-Entity Recognition
ner = pipeline("ner", grouped_entities=True)
output = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
print(output)
