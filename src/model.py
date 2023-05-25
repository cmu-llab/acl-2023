import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from special_tokens import *


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(TokenEmbedding, self).__init__()
        self._embedding = nn.Embedding(vocab_size, embedding_size)
        self._embedding_size = embedding_size

    def forward(self, tokens):
        return self._embedding(tokens.long()) * math.sqrt(self._embedding_size)


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, dropout, max_length):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, embedding_size, 2)* math.log(10000) / embedding_size)
        pos = torch.arange(0, max_length).reshape(max_length, 1)
        pos_embedding = torch.zeros((max_length, embedding_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self._dropout = nn.Dropout(dropout)
        self.register_buffer('_pos_embedding', pos_embedding)

    def forward(self, token_embedding, lens=None):
        if lens is None:  # each sequence is one sequence (target)
            return self._dropout(token_embedding + self._pos_embedding[:, :token_embedding.size(1), :token_embedding.size(2)])
        else:  # each sequence is a concatenation of multiple sequences (source)
            pos_embedding = []
            for _lens in lens:
                _pos_embedding = []
                for len in _lens:
                    _pos_embedding.append(self._pos_embedding[:, :len, :token_embedding.size(2)])  # positional encoding is applied per subsequence
                _pos_embedding = torch.cat(_pos_embedding, axis=1)  # subsequence-wise positional encodings concatenated
                pos_embedding.append(torch.squeeze(_pos_embedding, axis=0))
            pos_embedding = pad_sequence(pos_embedding, batch_first=True, padding_value=0)
            return self._dropout(token_embedding + pos_embedding)


class Model(nn.Module):
    def __init__(
        self,
        num_encoder_layers,
        num_decoder_layers,
        embedding_size,
        nhead,
        dim_feedforward,
        dropout,
        max_length,
        ipa_vocab,
        dialect_vocab
    ):
        super(Model, self).__init__()
        self._ipa_vocab = ipa_vocab
        self._dialect_vocab = dialect_vocab

        model_dim = embedding_size  # transformer d_model

        self._transformer = nn.Transformer(
            d_model=model_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self._generator = nn.Linear(model_dim, len(ipa_vocab))

        self._ipa_embedding = TokenEmbedding(len(ipa_vocab), embedding_size)
        self._dialect_embedding = TokenEmbedding(len(dialect_vocab), model_dim)
        self._positional_encoding = PositionalEncoding(embedding_size, dropout, max_length)

    def forward(
        self,
        source,
        dialect,
        source_lens,
        target,
        source_mask,
        target_mask,
        source_padding_mask,
        target_padding_mask,
        memory_key_padding_mask
    ):
        source_embedding = self._positional_encoding(self._ipa_embedding(source), source_lens)        
        target_embedding = self._positional_encoding(self._ipa_embedding(target))
    
        dialect_embedding = self._dialect_embedding(dialect)

        # add dialect embedding into source embedding
        source_embedding += dialect_embedding

        memory = self._transformer.encoder(
            source_embedding,
            mask=source_mask,
            src_key_padding_mask=source_padding_mask
        )

        outs = self._transformer.decoder(
            target_embedding, memory,
            tgt_mask=target_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        return self._generator(outs)

    def encode(self, source, source_lens, dialect, source_mask, source_padding_mask):
        source_embedding = self._positional_encoding(self._ipa_embedding(source), source_lens)

        dialect_embedding = self._dialect_embedding(dialect)
        source_embedding += dialect_embedding
        memory = self._transformer.encoder(
            source_embedding,
            mask=source_mask,
            src_key_padding_mask=source_padding_mask
        )
        return memory

    def decode(self, target, memory, target_mask):
        target_embedding = self._positional_encoding(self._ipa_embedding(target))

        return self._transformer.decoder(target_embedding, memory, tgt_mask=target_mask)
