from transformers import pipeline

question_answer = pipeline("question-answering")
output = question_answer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)
print(output)
