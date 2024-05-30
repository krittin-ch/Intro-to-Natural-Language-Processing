from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
output = translator("J'étudiais au SIIT quand j'avais 20 ans. J'aime beaucoup manger du fromage au petit-déjeuner.")
print(output)