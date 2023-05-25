import json
import random
import pickle
from collections import namedtuple

import numpy as np
import torch

from data import Vocab


def build_vocab(fpath):
    vocab = set()
    dialects = set()

    with open(fpath, 'rb') as fin:
        data = pickle.load(fin)
        languages = data[0]
        for char, entry in data[1].items():
            target = entry['protoform'][languages[0]]
            vocab.update(target)
            for dialect, source in entry['daughters'].items():
                vocab.update(source)
                dialects.add(dialect)

    ipa_vocab = Vocab(vocab)
    dialect_vocab = Vocab(dialects)
    print(f'ipa vocabulary: {len(ipa_vocab)}')
    print(f'dialect vocabulary: {len(dialect_vocab)}')

    return ipa_vocab, dialect_vocab


def load_config(conf_fpath):
    with open(conf_fpath) as fin:
        d = json.load(fin)
    conf = namedtuple('conf', d.keys())(**d)

    for k, v in conf._asdict().items():
        print(f'  * {k}: {v}')
    print()

    return conf


def dict_to_config(d):
    return namedtuple('conf', d.keys())(**d)


def get_edit_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


# source: https://github.com/neubig/minbert-assignment/blob/main/classifier.py
def seed_everything(seed=12345):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
