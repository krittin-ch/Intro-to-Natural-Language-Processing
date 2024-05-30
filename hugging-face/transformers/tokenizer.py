from transformers import BertTokenizer, AutoTokenizer

# Word-based tokenizer
text = 'Jim Henson was a puppeteer'
tokenized_text = text.split()
# ['Jim', 'Henson', 'was', 'a', 'puppeteer']

# Character-based tokenizer
tokenized_text = [char for char in text if not char.isspace()]
print(tokenized_text)
# ['J', 'i', 'm', 'H', 'e', 'n', 's', 'o', 'n', 'w', 'a', 's', 'a', 'p', 'u', 'p', 'p', 'e', 't', 'e', 'e', 'r']

# Subword tokenizer
# Subword tokenization algorithms rely on the principle that frequently used words should not be split into smaller subwords
# , but rare words should be decomposed into meaningful subwords.

# BERT tokenizer trained with the same checkpoint as BERT is done the same way as loading the model, except we use the BertTokenizer class
sequence = "Using a Transformer network is simple"
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
output = tokenizer(sequence)

'''
{'input_ids': [101, 7993, 170, 13809, 23763, 2443, 1110, 3014, 102],
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0],
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
'''

# AutoTokenizer class will grab the proper tokenizer class in the library based on the checkpoint name, and can be used directly with any checkpoint
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
output = tokenizer(sequence)

'''
{'input_ids': [101, 7993, 170, 13809, 23763, 2443, 1110, 3014, 102],
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0],
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
'''

# This tokenizer is a subword tokenizer: it splits the words until it obtains tokens that can be represented by its vocabulary.
tokens = tokenizer.tokenize(sequence)
# ['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']    

# The conversion to input_ids is handled by the convert_tokens_to_ids() tokenizer method
ids = tokenizer.convert_tokens_to_ids(tokens)
# [7993, 170, 13809, 23763, 2443, 1110, 3014]

decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)
