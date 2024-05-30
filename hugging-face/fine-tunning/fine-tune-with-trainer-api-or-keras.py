from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer
import numpy as np
import evaluate

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Prepare Dataset
raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training
# TrainingArguments class that will contain all the hyperparameters the Trainer will use for training and evaluation.
#  The only argument you have to provide is a directory where the trained model will be saved,
# as well as the checkpoints along the way. For all the rest, you can leave the defaults,
# which should work pretty well for a basic fine-tuning.
training_args = TrainingArguments("test-trainer")

# Define Model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# Define Trainer
# Define a Trainer by passing it all the objects constructed up to now
# — the model, the training_args, the training and validation datasets, our data_collator, and our tokenizer

'''

# We didn'  t tell the Trainer to evaluate during training by setting evaluation_strategy to either "steps" (evaluate every eval_steps) or "epoch" (evaluate at the end of each epoch).
training_args = TrainingArguments("test-trainer")


trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,    # default = DataCollatorWithPadding
    tokenizer=tokenizer,
)

# Fine-Tune
trainer.train()

predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)

pred = compute_metrics(predictions)
print(pred)

'''

# TrainingArguments with its evaluation_strategy set to "epoch" and a new model 
# — otherwise, we would just be continuing the training of the model we have already trained.
training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
