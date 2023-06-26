import pickle

import torch
import torch.nn as nn

from data.vocab import Vocab
from data.special_tokens import *


def create_mask(src, tgt, device):
    '''
    * src: batch of source index sequences [batch_size, num_dialects, seq_len]
    * tgt: batch of target index sequences [batch_size, seq_len]
    '''
    bs = src.shape[0]
    src_seq_len = src.shape[2]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).bool().to(device)
    src_padding_mask = (src == PAD_IDX)
    tgt_padding_mask = (tgt == PAD_IDX)

    src_mask = torch.zeros((src_seq_len, src_seq_len)).bool().to(device)

    memory_key_padding_mask = src_padding_mask.reshape((bs, -1))
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask


def build_vocab(train_fpath, lang_separators=False):
    '''
    * build ipa vocabulary and dialect vocabulary from training data
    '''
    vocab = []
    dialects = []

    with open(train_fpath, 'rb') as fin:
        langs, data = pickle.load(fin)
        for char, entry in data.items():
            target = entry['protoform']
            vocab += list(target.values())[0]
            for dialect, source in entry['daughters'].items():
                vocab += source
                dialects.append(dialect)

    # Vocab turns the list into a set-like dict
    ipa_vocab = Vocab(vocab)
    dialect_vocab = Vocab(dialects)
    dialect_vocab.protolang = langs[0]
    dialect_vocab.daughter_langs = langs[1:]
    print(f'ipa vocabulary: {len(ipa_vocab)}')
    print(f'dialect vocabulary: {len(dialect_vocab)}')

    if lang_separators:
        # note - the separators are already in the vocabulary
        # add daughter languages to the token vocabulary
        print("adding daughter languages to ipa_vocab")
        for lang in dialect_vocab.daughter_langs:
            ipa_vocab.add_token(lang)
        print(f'dialect vocabulary: {len(dialect_vocab)}')

        print("adding the protolang to dialect_vocab")
        dialect_vocab.add_token(dialect_vocab.protolang)
        # special tokens will belong to this separate language
        print("adding the separator language to dialect_vocab")
        dialect_vocab.add_token(SEPARATOR_LANG)
        print(f'dialect vocabulary: {len(dialect_vocab)}')

    return ipa_vocab, dialect_vocab
