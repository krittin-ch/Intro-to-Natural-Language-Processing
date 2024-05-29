import nltk
from nltk.corpus import conll2002

sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Despite the heavy rain, the match continued until the last minute.",
    "He asked, 'Do you know where the nearest train station is?'",
    "The algorithm optimizes the processing time by utilizing advanced machine learning techniques.",
    "In the serene valley, wildflowers bloom vibrantly under the golden sun.",
    "Artificial intelligence is transforming various industries rapidly.",
    "She enjoys reading books on a rainy afternoon.",
    "The chef prepared a delicious meal with fresh ingredients.",
    "Learning new languages can be challenging yet rewarding.",
    "The company announced its quarterly earnings last Monday."
]

text = sentences[0]
words = nltk.word_tokenize(text)

# POS tagging
pos_tags = nltk.pos_tag(words)

named_entities = nltk.ne_chunk(pos_tags)                # perform NER
iob_tags = nltk.chunk.tree2conlltags(named_entities)    # convert NER into cornll format


print('POS tags : \n', pos_tags)
print()
print('NER tags : \n', iob_tags)

# tagged_sentences = nltk.corpus.treebank.tagged_sents()
