from transformers import pipeline

# Text Generation
generator = pipeline("text-generation")
output = generator("In this course, we will teach you how to")

print(output)


generator = pipeline("text-generation", model="distilgpt2")
output = generator(
    "In this course, we will teach you how to",
    max_length = 30,
    num_return_sequences=2,
)

print(output)
