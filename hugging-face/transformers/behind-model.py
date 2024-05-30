from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch

# Training the sentimental analysis

# Preprocessing with a tokenizer
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"      # this is the default (pre-trained) model for sentimental analysis
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]

'''
Padding involves adding extra tokens (often zeros) to the input sequence to make it reach a predefined length.
Truncation involves cutting off part of the input sequence if it exceeds the predefined length. 
'''

inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")

'''
inputs =
{'input_ids': tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
          2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,
             0,     0,     0,     0,     0,     0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])}
'''
# from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)
# from_pretrained(cls, pretrained_model_name_or_path)
#   = from_pretrained(inputs['input_ids'], inputs['attention_mask'])
#   = from_pretrained(**inputs)

# Exclude model head and prediction
model = AutoModel.from_pretrained(checkpoint)

outputs = model(**inputs)               # the output from the hidden states of the model - not the actual model output
# print(outputs.last_hidden_state.shape)  # torch.Size([2, 16, 768])

# Excluse only prediction stage (include model head) - provide only raw score
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
# print(outputs.logits.shape) # torch.Size([2, 2])
# print(outputs.logits)

'''
tensor([[-1.5607,  1.6123],
        [ 4.1692, -3.3464]], grad_fn=<AddmmBackward>)

Our model predicted [-1.5607, 1.6123] for the first sentence and [ 4.1692, -3.3464] for the second one. 
Those are not probabilities but logits, the raw, unnormalized scores outputted by the last layer of the model.        
'''

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
# print(predictions)
'''
tensor([[4.0195e-02, 9.5980e-01],
        [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward0>)

The model predicted [0.0402, 0.9598] for the first sentence and [0.9995, 0.0005] for the second one.
'''

# Model labeling
# print(model.config.id2label)    # {0: 'NEGATIVE', 1: 'POSITIVE'}
'''
First sentence: NEGATIVE: 0.0402, POSITIVE: 0.9598
Second sentence: NEGATIVE: 0.9995, POSITIVE: 0.0005
'''