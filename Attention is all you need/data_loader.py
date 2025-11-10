from datasets import load_dataset
import pandas as pd
import spacy
from collections import Counter

# Load the Multi30k dataset
dataset = load_dataset("bentrevett/multi30k")
train_data = dataset['train']
test_data = dataset['test']
validation_data = dataset['validation']

# Spacy tokenizers
spacy_eng = spacy.load("en_core_web_sm")
spacy_ger = spacy.load("de_core_news_sm")

def tokenize_eng(text):
    if isinstance(text, list):
        text = text[0] if text else ""
    elif not isinstance(text, str):
        text = str(text)
    return [t.text.lower() for t in spacy_eng.tokenizer(text)]

def tokenize_ger(text):
    if isinstance(text, list):
        text = text[0] if text else ""
    elif not isinstance(text, str):
        text = str(text)
    return[t.text.lower() for t in spacy_ger.tokenizer(text)]

#Adding <sos> and <eos>

# <pad>: Padding token for sequence batching.
# <unk>: Unknown token for out-of-vocabulary words.
# <sos>: Start of Sequence
# <eos>: End of Sequence

def build_vocab(token_lists, max_size=10000,min_freq=2):
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    vocab={"<pad>":0,"<unk>":1,"<sos>":2,"<eos>":3}
    idx=4
    for word, freq in counter.most_common(max_size):
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab

src_vocab = build_vocab([tokenize_eng(data['en']) for data in train_data])
tgt_vocab = build_vocab([tokenize_ger(data['de']) for data in train_data])

class TranslationDaataset:
    def __init__(self, data, src_vocab, tgt_vocab):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        en_text = item['en']
        de_text = item['de']
        
        src = [self.src_vocab.get(tok, self.src_vocab["<unk>"]) for tok in tokenize_eng(en_text)]
        tgt = [self.tgt_vocab["<sos>"]] + [self.tgt_vocab.get(tok, self.tgt_vocab["<unk>"]) for tok in tokenize_ger(de_text)] + [self.tgt_vocab["<eos>"]]
        return src, tgt

def create_batches(dataset, batch_size=64):
    batches = []
    for i in range(0, len(dataset), batch_size):
        batch_src = []
        batch_tgt = []
        for j in range(i, min(i+batch_size, len(dataset))):
            src, tgt = dataset[j]
            batch_src.append(src)
            batch_tgt.append(tgt)
        batches.append((batch_src, batch_tgt))
    return batches

train_dataset = TranslationDaataset(train_data, src_vocab, tgt_vocab)
valid_dataset = TranslationDaataset(validation_data, src_vocab, tgt_vocab)
