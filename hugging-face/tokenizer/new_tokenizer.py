from datasets import load_dataset
from transformers import AutoTokenizer

# This can take a few minutes to load, so grab a coffee or tea while you wait!
raw_datasets = load_dataset("code_search_net", "python")
'''
DatasetDict({
    train: Dataset({
        features: ['repository_name', 'func_path_in_repository', 'func_name', 'whole_func_string', 'language', 'func_code_string', 'func_code_tokens', 'func_documentation_string', 'func_documentation_tokens', 'split_name', 'func_code_url'],
        num_rows: 412178
    })
    test: Dataset({
        features: ['repository_name', 'func_path_in_repository', 'func_name', 'whole_func_string', 'language', 'func_code_string', 'func_code_tokens', 'func_documentation_string', 'func_documentation_tokens', 'split_name', 'func_code_url'],
        num_rows: 22176
    })
    validation: Dataset({
        features: ['repository_name', 'func_path_in_repository', 'func_name', 'whole_func_string', 'language', 'func_code_string', 'func_code_tokens', 'func_documentation_string', 'func_documentation_tokens', 'split_name', 'func_code_url'],
        num_rows: 23107
    })
})
'''
'''
print(raw_datasets["train"][123456]["whole_func_string"])
def last_rate_limit(self):
        """
        A `dict` of the rate limit information returned in the most recent
        response, or `None` if no requests have been made yet.  The `dict`
        consists of all headers whose names begin with ``"RateLimit"`` (case
        insensitive).

        The DigitalOcean API specifies the following rate limit headers:

        :var string RateLimit-Limit: the number of requests that can be made
            per hour
        :var string RateLimit-Remaining: the number of requests remaining until
            the limit is reached
        :var string RateLimit-Reset: the Unix timestamp for the time when the
            oldest request will expire from rate limit consideration
        """
        if self.last_response is None:
            return None
        else:
            return {k:v for k,v in iteritems(self.last_response.headers)
                        if k.lower().startswith('ratelimit')}
'''

def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000]["whole_func_string"]
        for i in range(0, len(raw_datasets["train"]), 1000)
    )


training_corpus = (
    raw_datasets["train"][i : i + 1000]["whole_func_string"]
    for i in range(0, len(raw_datasets["train"]), 1000)
)

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''

tokens = old_tokenizer.tokenize(example)
'''
['def', 'Ġadd', '_', 'n', 'umbers', '(', 'a', ',', 'Ġb', '):','Ċ', 'Ġ', 
'Ġ', 'Ġ', 'Ġ"""', 'Add', 'Ġthe', 'Ġtwo', 'Ġnumbers', 'Ġ`', 'a', '`', 'Ġand', 
'Ġ`', 'b', '`', '."', '""', 'Ċ', 'Ġ', 'Ġ', 'Ġ', 'Ġreturn', 'Ġa', 'Ġ+', 'Ġb']
'''

# Trained token will be up t 52000 tokens
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)

example = """class LinearLayer():
    def __init__(self, input_size, output_size):
        self.weight = torch.randn(input_size, output_size)
        self.bias = torch.zeros(output_size)

    def __call__(self, x):
        return x @ self.weights + self.bias
    """
tokens = tokenizer.tokenize(example)
print(tokens)

tokenizer.save_pretrained("code-search-net-tokenizer")
