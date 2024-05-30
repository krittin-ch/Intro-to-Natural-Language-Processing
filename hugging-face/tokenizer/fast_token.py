from transformers import AutoTokenizer, pipeline
import numpy as np
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
encoding = tokenizer(example)
# print(type(encoding))
# print(encoding)     # <class 'transformers.tokenization_utils_base.BatchEncoding'>
'''
{
    'input_ids': [101, 1422, 1271, 1110, 156, 7777, 2497, 1394, 1105, 146, 1250, 1120, 20164, 10932, 10289, 1107, 6010, 119, 102], 
    'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
'''

# print(encoding.tokens())      # ['[CLS]', 'My', 'name', 'is', 'S', '##yl', '##va', '##in', 'and', 'I', 'work', 'at', 'Hu', '##gging', 'Face', 'in', 'Brooklyn', '.', '[SEP]']       
# print(encoding.word_ids())    #[None, 0, 1, 2, 3, 3, 3, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, None]

# start, end = encoding.word_to_chars(3)
# print(example[start:end])   # Sylvain

token_classifier = pipeline("token-classification")
tokens = token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
'''
[{'entity': 'I-PER', 'score': 0.99938285, 'index': 4, 'word': 'S', 'start': 11, 'end': 12}, 
{'entity': 'I-PER', 'score': 0.99815494, 'index': 5, 'word': '##yl', 'start': 12, 'end': 14}, 
{'entity': 'I-PER', 'score': 0.99590707, 'index': 6, 'word': '##va', 'start': 14, 'end': 16}, 
{'entity': 'I-PER', 'score': 0.99923277, 'index': 7, 'word': '##in', 'start': 16, 'end': 18},
{'entity': 'I-ORG', 'score': 0.9738931, 'index': 12, 'word': 'Hu', 'start': 33, 'end': 35}, 
{'entity': 'I-ORG', 'score': 0.976115, 'index': 13, 'word': '##gging', 'start': 35, 'end': 40}, 
{'entity': 'I-ORG', 'score': 0.9887976, 'index': 14, 'word': 'Face', 'start': 41, 'end': 45}, 
{'entity': 'I-LOC', 'score': 0.9932106, 'index': 16, 'word': 'Brooklyn', 'start': 49, 'end': 57}]
'''

token_classifier = pipeline("token-classification", aggregation_strategy="simple")
tokens = token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
'''
[{'entity_group': 'PER', 'score': 0.9981694, 'word': 'Sylvain', 'start': 11, 'end': 18},
 {'entity_group': 'ORG', 'score': 0.9796019, 'word': 'Hugging Face', 'start': 33, 'end': 45},
   {'entity_group': 'LOC', 'score': 0.9932106, 'word': 'Brooklyn', 'start': 49, 'end': 57}]
'''

from transformers import AutoTokenizer, AutoModelForTokenClassification

model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
inputs = tokenizer(example, return_tensors="pt")
outputs = model(**inputs)

import torch

probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
predictions = outputs.logits.argmax(dim=-1)[0].tolist()
# print(predictions)              # [0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 6, 6, 6, 0, 8, 0, 0]
# print(model.config.id2label)    # {0: 'O', 1: 'B-MISC', 2: 'I-MISC', 3: 'B-PER', 4: 'I-PER', 5: 'B-ORG', 6: 'I-ORG', 7: 'B-LOC', 8: 'I-LOC'}

results = []
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
tokens = inputs_with_offsets.tokens()
offsets = inputs_with_offsets["offset_mapping"]

idx = 0
while idx < len(predictions):
    pred = predictions[idx]
    label = model.config.id2label[pred]
    if label != "O":
        # Remove the B- or I-
        label = label[2:]
        start, _ = offsets[idx]

        # Grab all the tokens labeled with I-label
        all_scores = []
        while (
            idx < len(predictions)
            and model.config.id2label[predictions[idx]] == f"I-{label}"
        ):
            all_scores.append(probabilities[idx][pred])
            _, end = offsets[idx]
            idx += 1

        # The score is the mean of all the scores of the tokens in that grouped entity
        score = np.mean(all_scores).item()
        word = example[start:end]
        results.append(
            {
                "entity_group": label,
                "score": score,
                "word": word,
                "start": start,
                "end": end,
            }
        )
    idx += 1

print(results)
'''
[{'entity_group': 'PER', 'score': 0.9981694370508194, 'word': 'Sylvain', 'start': 11, 'end': 18}, 
{'entity_group': 'ORG', 'score': 0.9796018997828165, 'word': 'Hugging Face', 'start': 33, 'end': 45}, 
{'entity_group': 'LOC', 'score': 0.9932106137275696, 'word': 'Brooklyn', 'start': 49, 'end': 57}]
'''
'''
[{'entity': 'I-PER', 'score': 0.9993828535079956, 'word': 'S', 'start': 11, 'end': 12}, 
{'entity': 'I-PER', 'score': 0.9981549382209778, 'word': '##yl', 'start': 12, 'end': 14}, 
{'entity': 'I-PER', 'score': 0.995907187461853, 'word': '##va', 'start': 14, 'end': 16}, 
{'entity': 'I-PER', 'score': 0.9992327690124512, 'word': '##in', 'start': 16, 'end': 18}, 
{'entity': 'I-ORG', 'score': 0.9738931059837341, 'word': 'Hu', 'start': 33, 'end': 35}, 
{'entity': 'I-ORG', 'score': 0.9761149883270264, 'word': '##gging', 'start': 35, 'end': 40}, 
{'entity': 'I-ORG', 'score': 0.9887976050376892, 'word': 'Face', 'start': 41, 'end': 45}, 
{'entity': 'I-LOC', 'score': 0.9932106137275696, 'word': 'Brooklyn', 'start': 49, 'end': 57}]
'''