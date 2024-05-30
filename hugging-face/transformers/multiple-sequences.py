import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.tensor([ids])
'''
print("Input IDs:", input_ids)
Input IDs: tensor([[ 1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,  2607,
          2026,  2878,  2166,  1012]])
'''
output = model(input_ids)
'''
print("Logits:", output.logits)
Logits: tensor([[-2.7276,  2.8789]], grad_fn=<AddmmBackward0>)
'''


# Padding Inputs

'''
# Here, the input dimensions are not identical
batched_ids = [
    [200, 200, 200],
    [200, 200]
]

padding_id = 100

batched_ids = [
    [200, 200, 200],
    [200, 200, padding_id],
]
'''

# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

# print(model(torch.tensor(sequence1_ids)).logits)    # tensor([[ 1.5694, -1.3895]], grad_fn=<AddmmBackward0>)
# print(model(torch.tensor(sequence2_ids)).logits)    # tensor([[ 0.5803, -0.4125]], grad_fn=<AddmmBackward0>)/
# print(model(torch.tensor(batched_ids)).logits)
'''
tensor([[ 1.5694, -1.3895],
        [ 1.3374, -1.2163]], grad_fn=<AddmmBackward0>)
'''
# print("Padding ID : "tokenizer.pad_token_id)   # Padding ID : 0

# Attention Masks

'''
Attention masks are tensors with the exact same shape as the input IDs tensor, 
filled with 0s and 1s: 1s indicate the corresponding tokens should be attended to, 
and 0s indicate the corresponding tokens should not be attended to 
(i.e., they should be ignored by the attention layers of the model).
'''

batched_ids = [
    [200, 200, 200],
    [200, 200, 1000],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))

print(outputs.logits)
'''

tensor([[ 1.5694, -1.3895],
        [ 0.5803, -0.4125]], grad_fn=<AddmmBackward0>)
'''


# Longer sequences
'''
With Transformer models, there is a limit to the lengths of the sequences we can pass the models. 
Most models handle sequences of up to 512 or 1024 tokens, and will crash when asked to process longer sequences. There are two solutions to this problem:

[1] Use a model with a longer supported sequence length.
[2] Truncate your sequences.

Models have different supported sequence lengths, and some specialize in handling very long sequences. Longformer is one example, and another is LED. If you’re working on a task that requires very long sequences, we recommend you take a look at those models.

Otherwise, we recommend you truncate your sequences by specifying the max_sequence_length parameter:

sequence = sequence[:max_sequence_length]
'''
