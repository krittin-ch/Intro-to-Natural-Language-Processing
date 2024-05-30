from transformers import BertConfig, BertModel, AutoTokenizer
import torch

# # Building the config
# config = BertConfig()

'''
BertConfig {
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.41.1",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 30522
}
'''


# # Building the model from the config
# model = BertModel(config)
# # Model is randomly initialized!
# # The model can be used in this state, but it will output gibberish; it needs to be trained first.

# Obtain the pre-trained model
checkpoint = "bert-base-cased"
model = BertModel.from_pretrained(checkpoint)

sequences = ["Hello!", "Cool.", "Nice!"]

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

encoded_sequences = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
'''
{'input_ids': tensor([[  101,  8667,   106,   102],
        [  101, 13297,   119,   102],
        [  101,  8835,   106,   102]]), 'token_type_ids': tensor([[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]])}
'''
model_inputs = encoded_sequences['input_ids']

output = model(model_inputs)    # This model can accept various different arguments (inputs), only the input_ids is necessary
'''
BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=tensor([[[ 0.6283,  0.2166,  0.5605,  ...,  0.0136,  0.6158, -0.1712],
         [ 0.6108, -0.2253,  0.9263,  ..., -0.3028,  0.4500, -0.0714],
         [ 0.8040,  0.1809,  0.7076,  ..., -0.0685,  0.4837, -0.0774],
         [ 1.3290,  0.2360,  0.4567,  ...,  0.1509,  0.9621, -0.4841]],

        [[ 0.3128,  0.1718,  0.2099,  ..., -0.0721,  0.4919, -0.1383],
         [ 0.1545, -0.3757,  0.7187,  ..., -0.3130,  0.2822,  0.1883],
         [ 0.4123,  0.3721,  0.5484,  ...,  0.0788,  0.5681, -0.2757],
         [ 0.8356,  0.3964, -0.4121,  ...,  0.1838,  1.6365, -0.4806]],

        [[ 0.5399,  0.2564,  0.2511,  ..., -0.1760,  0.6063, -0.1803],
         [ 0.2609, -0.3164,  0.5548,  ..., -0.3439,  0.3909,  0.0900],
         [ 0.5161,  0.0721,  0.5606,  ...,  0.0077,  0.3685, -0.2272],
         [ 0.6560,  0.8475, -0.1606,  ..., -0.0468,  1.6309, -0.5047]]],
       grad_fn=<NativeLayerNormBackward0>), pooler_output=tensor([[-0.7105,  0.4876,  0.9999,  ...,  1.0000, -0.9179,  0.9894],
        [-0.7731,  0.5619,  1.0000,  ...,  1.0000, -0.8397,  0.9944],
        [-0.7594,  0.5645,  1.0000,  ...,  1.0000, -0.9015,  0.9969]],
       grad_fn=<TanhBackward0>), hidden_states=None, past_key_values=None, attentions=None, cross_attentions=None)
'''
