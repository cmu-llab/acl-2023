import math

import torch
import torch.nn as nn

from data.special_tokens import *


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(TokenEmbedding, self).__init__()
        self._embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD_IDX)
        self._embedding_size = embedding_size

    def forward(self, tokens):
        return self._embedding(tokens.long()) * math.sqrt(self._embedding_size)


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, dropout, max_length):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, embedding_size, 2) * math.log(10000) / embedding_size)
        pos = torch.arange(0, max_length).reshape(max_length, 1)
        pos_embedding = torch.zeros((max_length, embedding_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0).unsqueeze(0)

        self._dropout = nn.Dropout(dropout)
        self.register_buffer('_pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        '''
        * token_embedding: [batch_size, n_dialects, seq_len, embedding_dim]
        '''
        return self._dropout(token_embedding + self._pos_embedding[:, :, token_embedding.size(2)])
