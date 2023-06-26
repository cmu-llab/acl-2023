import pickle

import torch
from torch.nn.utils.rnn import pad_sequence

from data.special_tokens import *


class DatasetBase(torch.utils.data.Dataset):
    def __init__(self, fpath, ipa_vocab, dialect_vocab, data_shape, skip_daughter_tone, skip_protoform_tone):
        self.ipa_vocab = ipa_vocab
        self.dialect_vocab = dialect_vocab
        self.data_shape = data_shape
        self.skip_daughter_tone = skip_daughter_tone
        self.skip_protoform_tone = skip_protoform_tone

        # load dataset
        X = []
        Y = []
        with open(fpath, 'rb') as fin:
            langs, data = pickle.load(fin)
            for cognate, entry in data.items():
                X.append(entry['daughters'])
                Y.append(entry['protoform'])

        self.num_daughters = len(langs) - 1
        self.protolang = langs[0]
        self.daughter_langs = langs[1:]

        self.X = X
        self.Y = Y

        self.length = len(self.X)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        raise NotImplementedError()

    def collate_fn(self, batch):
        raise NotImplementedError()


class DatasetConcat(DatasetBase):
    def __init__(self, lang_separators=False, **kwargs):
        super(DatasetConcat, self).__init__(**kwargs)

        self.lang_separators = lang_separators
        if lang_separators:
            assert self.dialect_vocab.get_idx(SEPARATOR_LANG) != UNK_IDX
            assert self.dialect_vocab.get_idx(self.protolang) != UNK_IDX
            for lang in self.daughter_langs:
                assert self.dialect_vocab.get_idx(lang) != UNK_IDX

    '''
    * used by Meloni et al (2021)
    '''
    def __getitem__(self, idx):
        '''
        * concatenate daughter sequences
        '''
        x_sequences, x_dialects, x_lens = [], [], []
        for dialect, sequence in self.X[idx].items():
            if self.skip_daughter_tone:
                sequence = sequence[:-1]
            dialect_seq = [dialect] * len(sequence)

            if self.lang_separators:
                sequence = [SPECIAL_TOKENS[COGNATE_SEP_IDX], dialect, SPECIAL_TOKENS[LANG_SEP_IDX]] + sequence
                dialect_seq = [SEPARATOR_LANG] * 3 + dialect_seq

            x_sequences.extend([self.ipa_vocab.get_idx(tkn) for tkn in sequence])
            x_dialects.extend([self.dialect_vocab.get_idx(dialect) for dialect in dialect_seq])
            x_lens.append(len(sequence))

        # example: BOS * French : croître * Italian : crescere * Spanish : crecer * Portuguese : crecer * Romanian : crește * EOS
        #   starts and ends with BOS and EOS
        #   COGNATE_SEP_IDX separates daughter cognates from each other
        #   the language tokens are one token
        #   LANG_SEP_IDX separates the language token from the actual cognate sequence
        x_sequences = [BOS_IDX] + x_sequences + [COGNATE_SEP_IDX, EOS_IDX]
        x_sequences = torch.tensor(x_sequences, dtype=torch.long)  # concatenation of all daughter sequences
        x_dialects = [self.dialect_vocab.get_idx(SEPARATOR_LANG)] + x_dialects + \
                     [self.dialect_vocab.get_idx(SEPARATOR_LANG)] * 2
        x_dialects = torch.tensor(x_dialects, dtype=torch.long)  # dialect index for each token in the concatenated sequence
        # x_lens doesn't include the BOS_IDX, COGNATE_SEP_IDX, EOS_IDX
        x_lens = torch.tensor(x_lens, dtype=torch.long)  # lengths of each daughter sequence

        y_sequence = list(self.Y[idx].values())[0]
        if self.skip_protoform_tone:
            y_sequence = y_sequence[:-1]
        y_sequence = [BOS_IDX] + [self.ipa_vocab.get_idx(tkn) for tkn in y_sequence] + [EOS_IDX]
        y_sequence = torch.tensor(y_sequence, dtype=torch.long)

        return x_sequences, x_dialects, x_lens, y_sequence

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


class DatasetStack(DatasetBase):
    def __getitem__(self, idx):
        '''
        * stack daughter sequences
        '''
        x_sequences, x_dialects = [], []
        for dialect, sequence in self.X[idx].items():
            if self.skip_daughter_tone:
                sequence = sequence[:-1]  # tone must be a single token at the end of the sequence
            sequence = [SPECIAL_TOKENS[BOS_IDX]] + sequence + [SPECIAL_TOKENS[EOS_IDX]]

            x_sequences.append(torch.tensor([self.ipa_vocab.get_idx(tkn) for tkn in sequence], dtype=torch.long))
            x_dialects.append(torch.tensor([self.dialect_vocab.get_idx(dialect)], dtype=torch.long))

        # pseudo-padding for missing daughters
        for _ in range(self.num_daughters - len(self.X[idx])):
            '''
            < note about pseudo-padding >
            * If an entire sequence is just paddings (without even bos or eos)
            * the entire sequence will be masked and the output will be nan for all tokens
            '''
            if self.data_shape == 'stack':
                empty_sequence = [BOS_IDX, EOS_IDX]
            else:
                empty_sequence = []

            x_sequences.append(torch.tensor(empty_sequence, dtype=torch.long))
            x_dialects.append(torch.tensor([UNK_IDX]))  # TODO: should this be PAD_IDX instead?

        x_sequences = pad_sequence(x_sequences, batch_first=True, padding_value=PAD_IDX)  # [num_daughters, max_sequence_length]
        x_dialects = torch.stack(x_dialects)  # [num_daughters, 1]

        y_sequence = list(self.Y[idx].values())[0]
        if self.skip_protoform_tone:
            y_sequence = y_sequence[:-1]
        y_sequence = [BOS_IDX] + [self.ipa_vocab.get_idx(tkn) for tkn in y_sequence] + [EOS_IDX]
        y_sequence = torch.tensor(y_sequence, dtype=torch.long)

        return x_sequences, x_dialects, y_sequence

    def _pad_sequence_2d(self, sequences, batch_first=False, padding_value=PAD_IDX):
        '''
        * `sequences`: list of stacked sequences.
        * Each stack was padded to a different length in `__getitem__()`. Add padding so all stacks have the same sequence length.
        '''
        assert batch_first, "currently does not support batch_first==False"

        max_inner_len = max(s.shape[1] for s in sequences)
        # add dummy element to make sure pad_sequence pads to the maximum length
        dummy = torch.zeros(max_inner_len)
        padded_1d = [pad_sequence(list(s)+[dummy], batch_first=batch_first, padding_value=padding_value)[:-1]  # remove dummy after padding
                     for s in sequences]
        return torch.stack(padded_1d)

    def collate_fn(self, batch):
        X_sequences, X_dialects = [], []
        Y_sequences = []

        for x_sequence, x_dialects, y_sequence in batch:
            X_sequences.append(x_sequence)
            X_dialects.append(x_dialects)

            Y_sequences.append(y_sequence)

        X_sequences = self._pad_sequence_2d(X_sequences, batch_first=True, padding_value=PAD_IDX)
        X_dialects = pad_sequence(X_dialects, batch_first=True, padding_value=PAD_IDX)
        Y_sequences = pad_sequence(Y_sequences, batch_first=True, padding_value=PAD_IDX)

        return X_sequences, X_dialects, Y_sequences
