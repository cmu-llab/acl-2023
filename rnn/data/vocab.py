from data.special_tokens import *
        

class Vocab:
    def __init__(self, tokens):
        # dict maintains insertion order as of Python 3.7+, enable reproducibility
        tokens = dict.fromkeys(tokens)
        
        self.v2i = {special_token: idx for idx, special_token in enumerate(SPECIAL_TOKENS)}

        for idx, token in enumerate(tokens):
            self.v2i[token] = idx + len(SPECIAL_TOKENS)

        self.i2v = {v: k for k, v in self.v2i.items()}
        assert len(self.v2i) == len(self.i2v)

    def to_tokens(self, index_sequence, remove_special=True):
        '''
        * convert indices to tokens
        '''
        ret = []
        for idx in index_sequence:
            idx = idx.item()
            if remove_special:
                if idx in {UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX}:
                    continue
            ret.append(self.i2v[idx])

        return ret

    def get_idx(self, v):
        return self.v2i.get(v, UNK_IDX)

    def __len__(self):
        return len(self.v2i)

    def __getitem__(self, idx):
        return self.i2v.get(idx, self.i2v[UNK_IDX])

    def __iter__(self):
        for idx, tkn in sorted(self.i2v.items(), key=lambda x: x[0]):
            yield idx, tkn

    def add_token(self, token):
        index = len(self.v2i)
        self.v2i[token] = index
        self.i2v[index] = token
