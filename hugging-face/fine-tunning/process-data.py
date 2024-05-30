import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

from datasets import load_dataset

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


# The dataset consists of 5,801 pairs of sentences, with a label indicating if they are paraphrases or not
raw_datasets = load_dataset("glue", "mrpc")

raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset[0])

'''
raw_train_dataset.features
# {'sentence1': Value(dtype='string', id=None), 
# 'sentence2': Value(dtype='string', id=None),
# 'label': ClassLabel(names=['not_equivalent', 'equivalent'], id=None),
# 'idx': Value(dtype='int32', id=None)}
'''

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

'''
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)
'''
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
'''
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 1725
    })
})
'''

# Dynamic Padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Here, we remove the columns idx, sentence1, and sentence2 as they won’t be needed and contain strings
# (and we can’t create tensors with strings) and have a look at the lengths of each entry in the batch:

samples = tokenized_datasets["train"][:8]

samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}

# [len(x) for x in samples["input_ids"]] --> [50, 59, 47, 67, 59, 50, 62, 32]

batch = data_collator(samples)
'''
{k: v.shape for k, v in batch.items()}
--> {'input_ids': torch.Size([8, 67]), 
    'token_type_ids': torch.Size([8, 67]),
    'attention_mask': torch.Size([8, 67]), 
    'labels': torch.Size([8])}
'''


# optimizer = AdamW(model.parameters())
# loss = model(**batch).loss
# loss.backward()
# optimizer.step()
