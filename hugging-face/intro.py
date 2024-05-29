from transformers import pipeline
'''

# Sentimental Analysis (feelings) : positive, negative, neutral
classifier = pipeline("sentiment-analysis")     # download sentimental analysis model

output = classifier("I've been waiting for a HuggingFace course my whole life.")
# print(output)

# Zero-Shot Classification : 

classifier = pipeline("zero-shot-classification")

output = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
# print(output)

# Text Generation
generator = pipeline("text-generation")
output = generator("In this course, we will teach you how to")
print(output)
print()

generator = pipeline("text-generation", model="distilgpt2")

output = generator(
    "In this course, we will teach you how to",
    max_length = 30
    num_return_sequences=2,
)

print(output)

'''

# Mask Filling : FIll the blank
unmasker = pipeline("fill-mask")
output = unmasker("This is the <mask> opportunity to study at this top university in the united state.", top_k=2)
print(output)

'''
ner = pipeline("ner", grouped_entities=True)
output = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
print(output)

question_answerer = pipeline("question-answering")
output = question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)
print(output)

summarizer = pipeline("summarization")
output = summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
"""
)
print(output)

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
output = translator("J'étudiais au SIIT quand j'avais 20 ans. J'aime beaucoup manger du fromage au petit-déjeuner.")
print(output)
'''