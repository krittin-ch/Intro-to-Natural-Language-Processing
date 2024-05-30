corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

from collections import defaultdict

word_freqs = defaultdict(int)

for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

print(word_freqs)

alphabet = []

for word in word_freqs.keys():
    for letter in word:
        if letter not in alphabet:
            alphabet.append(letter)
alphabet.sort()

print(alphabet)

# special tokens used by the model at the beginning of that vocabulary.
# In the case of GPT-2, the only special token is "<|endoftext|>":
vocab = ["<|endoftext|>"] + alphabet.copy()
splits = {word: [c for c in word] for word in word_freqs.keys()}    # 'This': ['T', 'h', 'i', 's'

def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs

pair_freqs = compute_pair_freqs(splits)
'''
defaultdict(<class 'int'>, {('T', 'h'): 3, ('h', 'i'): 3, ('i', 's'): 5, ('Ġ', 'i'): 2, 
('Ġ', 't'): 7, ('t', 'h'): 3, ('h', 'e'): 2, ('Ġ', 'H'): 1, ('H', 'u'): 1, ('u', 'g'): 1,
('g', 'g'): 1, ('g', 'i'): 1, ('i', 'n'): 2, ('n', 'g'): 1, ('Ġ', 'F'): 1, ('F', 'a'): 1, 
('a', 'c'): 1, ('c', 'e'): 1, ('Ġ', 'C'): 1, ('C', 'o'): 1, ('o', 'u'): 3, ('u', 'r'): 1,
('r', 's'): 2, ('s', 'e'): 3, ('Ġ', 'c'): 1, ('c', 'h'): 1, ('h', 'a'): 1, ('a', 'p'): 1, 
('p', 't'): 1, ('t', 'e'): 2, ('e', 'r'): 5, ('Ġ', 'a'): 5, ('a', 'b'): 2, ('b', 'o'): 1, 
('u', 't'): 1, ('t', 'o'): 4, ('o', 'k'): 3, ('k', 'e'): 3, ('e', 'n'): 4, ('n', 'i'): 2, 
('i', 'z'): 2, ('z', 'a'): 1, ('a', 't'): 2, ('t', 'i'): 2, ('i', 'o'): 2, ('o', 'n'): 2, 
('Ġ', 's'): 3, ('e', 'c'): 1, ('c', 't'): 1, ('s', 'h'): 1, ('h', 'o'): 2, ('o', 'w'): 2, 
('w', 's'): 1, ('e', 'v'): 1, ('v', 'e'): 1, ('r', 'a'): 3, ('a', 'l'): 2, ('z', 'e'): 1, 
('l', 'g'): 1, ('g', 'o'): 1, ('o', 'r'): 1, ('r', 'i'): 1, ('i', 't'): 1, ('h', 'm'): 1, 
('m', 's'): 1, ('H', 'o'): 1, ('o', 'p'): 1, ('p', 'e'): 1, ('e', 'f'): 1, ('f', 'u'): 1, 
('u', 'l'): 1, ('l', 'l'): 2, ('l', 'y'): 1, ('Ġ', 'y'): 1, ('y', 'o'): 1, ('Ġ', 'w'): 1, 
('w', 'i'): 1, ('i', 'l'): 1, ('Ġ', 'b'): 1, ('b', 'e'): 1, ('b', 'l'): 1, ('l', 'e'): 1, 
('Ġ', 'u'): 1, ('u', 'n'): 1, ('n', 'd'): 3, ('d', 'e'): 1, ('s', 't'): 1, ('t', 'a'): 1, 
('a', 'n'): 2, ('Ġ', 'h'): 1, ('e', 'y'): 1, ('a', 'r'): 1, ('r', 'e'): 1, ('t', 'r'): 1, 
('a', 'i'): 1, ('n', 'e'): 2, ('e', 'd'): 1, ('Ġ', 'g'): 1, ('g', 'e'): 1, ('n', 's'): 1})
'''
for i, key in enumerate(pair_freqs.keys()):
    print(f"{key}: {pair_freqs[key]}")
    if i >= 5:
        break

best_pair = ""
max_freq = None

for pair, freq in pair_freqs.items():
    if max_freq is None or max_freq < freq:
        best_pair = pair
        max_freq = freq

print(best_pair, max_freq)

merges = {("Ġ", "t"): "Ġt"}
vocab.append("Ġt")

def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits

splits = merge_pair("Ġ", "t", splits)
print(splits["Ġtrained"])

vocab_size = 50

while len(vocab) < vocab_size:
    pair_freqs = compute_pair_freqs(splits)
    best_pair = ""
    max_freq = None
    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    splits = merge_pair(*best_pair, splits)
    merges[best_pair] = best_pair[0] + best_pair[1]
    vocab.append(best_pair[0] + best_pair[1])

print(merges)

print(vocab)


def tokenize(text):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    splits = [[l for l in word] for word in pre_tokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split

    return sum(splits, [])

print(tokenize("This is not a token."))