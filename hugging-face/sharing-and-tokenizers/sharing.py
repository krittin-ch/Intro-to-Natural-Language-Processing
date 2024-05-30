from huggingface_hub import notebook_login
# notebook_login()

from transformers import TrainingArguments, AutoModelForMaskedLM, AutoTokenizer

# Add push_to_hub = True to upload
# training_args = TrainingArguments(
#     "bert-finetuned-mrpc", save_strategy="epoch", push_to_hub=True
# ) 

checkpoint = "camembert-base"

model = AutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model.push_to_hub("dummy-model")
tokenizer.push_to_hub("dummy-model")


from huggingface_hub import create_repo

# create_repo("dummy-model")
# 