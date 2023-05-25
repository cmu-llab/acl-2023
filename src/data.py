import pickle

import torch
from torch.nn.utils.rnn import pad_sequence

from special_tokens import *


class Vocab:
    def __init__(self, tokens):
        tokens = set(tokens)
        
        self.v2i = {special_token: idx for idx, special_token in enumerate(SPECIAL_TOKENS)}

        for idx, token in enumerate(tokens):
            self.v2i[token] = idx + len(SPECIAL_TOKENS)

        self.i2v = {v: k for k, v in self.v2i.items()}
        assert len(self.v2i) == len(self.i2v)

    def to_string(self, index_sequence, remove_special=True, return_list=False):
        '''
        returns string representation of index sequence
        '''
        ret = []
        for idx in index_sequence:
            idx = idx.item()
            if remove_special:
                if idx in {UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX}:
                    continue
            ret.append(self.i2v[idx])
        if return_list:
            return ret
        else:
            return ''.join(ret)

    def get_idx(self, v):
        return self.v2i.get(v, UNK_IDX)

    def __len__(self):
        return len(self.v2i)

    def __getitem__(self, idx):
        return self.i2v.get(idx, self.i2v[UNK_IDX])

    def __iter__(self):
        for idx, tkn in sorted(self.i2v.items(), key=lambda x: x[0]):
            yield idx, tkn


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, ipa_vocab, dialect_vocab):
        self.ipa_vocab = ipa_vocab
        self.dialect_vocab = dialect_vocab
        
        X = []
        Y = []
        with open(data_path, 'rb') as fin:
            data = pickle.load(fin)
            languages = data[0]
            for char, entry in data[1].items():
                X.append(entry['daughters'])
                Y.append(entry['protoform'][languages[0]])
        self.X = X
        self.Y = Y

        assert len(self.X) == len(self.Y)
        self.length = len(self.X)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x_sequence, x_dialects, x_lens = [], [], []
        for dialect, sequence in self.X[idx].items():
            sequence = [SPECIAL_TOKENS[BOS_IDX]] + sequence + [SPECIAL_TOKENS[EOS_IDX]]
            x_sequence.append(torch.tensor([self.ipa_vocab.get_idx(tkn) for tkn in sequence], dtype=torch.long))
            x_dialects.extend([self.dialect_vocab.get_idx(dialect)] * len(sequence))
            x_lens.append(len(sequence))

        x_sequence = torch.cat(x_sequence)  # concatenation of sequence of token indices
        x_dialects = torch.tensor(x_dialects, dtype=torch.long)  # set of dialect indices
        x_lens = torch.tensor(x_lens, dtype=torch.long)  # list of lengths for each dialect

        y_sequence = torch.tensor(
            [BOS_IDX] + [self.ipa_vocab.get_idx(tkn) for tkn in self.Y[idx]] + [EOS_IDX],
            dtype=torch.long)

        return x_sequence, x_dialects, x_lens, y_sequence

    def collate_fn(self, batch):
        X_sequences, X_dialects, X_lens = [], [], []
        Y_sequences = []

        for x_sequence, x_dialects, x_lens, y_sequence in batch:
            X_sequences.append(x_sequence)
            X_dialects.append(x_dialects)
            X_lens.append(x_lens)

            Y_sequences.append(y_sequence)

        X_sequences = pad_sequence(X_sequences, batch_first=True, padding_value=PAD_IDX)
        X_dialects = pad_sequence(X_dialects, batch_first=True, padding_value=PAD_IDX)
        X_lens = pad_sequence(X_lens, batch_first=True, padding_value=0)

        Y_sequences = pad_sequence(Y_sequences, batch_first=True, padding_value=PAD_IDX)

        return X_sequences, X_dialects, X_lens, Y_sequences
