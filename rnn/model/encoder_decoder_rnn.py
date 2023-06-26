import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from data.special_tokens import *


class Embedding(nn.Module):
    """Feature extraction:
    Looks up embeddings for a source character and the language,
    concatenates the two,
    then passes through a linear layer
    """
    def __init__(self, embedding_dim, num_ipa_tokens, num_langs):
        super(Embedding, self).__init__()
        self.char_embeddings = nn.Embedding(num_ipa_tokens, embedding_dim)
        self.lang_embeddings = nn.Embedding(num_langs, embedding_dim)
        # map concatenated source and language embedding to 1 embedding
        self.fc = nn.Linear(2 * embedding_dim, embedding_dim)

    def forward(self, char_indices, lang_indices):
        # input: (batch size, L)

        # both result in (batch size, L, E), where L is the length of the entire cognate set
        chars_embedded = self.char_embeddings(char_indices)
        lang_embedded = self.lang_embeddings(lang_indices)

        # concatenate the tensors to form one long embedding then map down to regular embedding size
        return self.fc(torch.cat((chars_embedded, lang_embedded), dim=-1))


class MLP(nn.Module):
    """
    Multi-layer perceptron to generate logits from the decoder state
    """
    def __init__(self, hidden_dim, feedforward_dim, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, 2 * feedforward_dim)
        self.fc2 = nn.Linear(2 * feedforward_dim, feedforward_dim)
        self.fc3 = nn.Linear(feedforward_dim, output_size, bias=False)

    # no need to perform softmax because CrossEntropyLoss does the softmax for you
    def forward(self, decoder_state):
        h = F.relu(self.fc1(decoder_state))
        scores = self.fc3(F.relu(self.fc2(h)))
        return scores


class Attention(nn.Module):
    def __init__(self, hidden_dim, embedding_dim):
        super(Attention, self).__init__()
        self.W_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_c_s = nn.Linear(embedding_dim, hidden_dim, bias=False)

    def forward(self, query, keys, encoded_input):
        # query: decoder state. [batch_size, 1, H]
        # keys: encoder states. [batch_size, L, H]
        query = self.W_query(query)
        # attention to calculate similarity between the query and each key
        # the query is broadcast over the sequence dimension (L)
        # scores: [batch_size, L, 1]
        scores = torch.matmul(keys, query.transpose(1, 2))
        # TODO: set padded values' attention weights to negative infinity BEFORE softmax

        # softmax to get a probability distribution over the L encoder states
        weights = F.softmax(scores, dim=-2)

        # weights: [batch_size, L, 1]
        # encoded_input: [batch_size, L, E] -> [batch_size, L, H]
        # keys: [batch_size, L, H]
        # values = keys + encoded_input (residual connection). [batch_size, L, H]
        # first weight each value vector by broadcasting weights to each hidden dim. results in [batch_size, L, H]
        values = self.W_c_s(encoded_input) + self.W_key(keys)
        weighted_states = weights * values
        # get a linear combination (weighted sum) of the value vectors. results in [batch_size, H]
        weighted_states = weighted_states.sum(dim=-2)

        return weighted_states


class EncoderDecoderRNN(nn.Module):
    """
    Encoder-decoder architecture
    """
    def __init__(self,
                 ipa_vocab,
                 dialect_vocab,
                 num_encoder_layers,
                 num_decoder_layers,
                 dropout,
                 feedforward_dim,
                 embedding_dim,
                 model_size):
        super(EncoderDecoderRNN, self).__init__()
        # vocabulary of IPA tokens shared among all languages
        self.ipa_vocab = ipa_vocab
        # vocabulary of language names, which includes the proto-language and the separator (punctuation) language in Meloni et al (2021)
        self.lang_vocab = dialect_vocab
        self.protolang = dialect_vocab.protolang

        # TODO: ensure that ipa_vocab is indeed C2I and langs is dialect_vocab - any discrepancies between here and Meloni?

        # share embedding across all languages, including the proto-language
        self.embeddings = Embedding(embedding_dim, len(self.ipa_vocab), len(self.lang_vocab))
        # have separate embedding for the language
        # technically, most of the vocab is not used in the separator embedding
        # since the separator tokens and the character embeddings are disjoint, put them all in the same matrix

        self.dropout = nn.Dropout(dropout)
        self.encoder_rnn = nn.GRU(input_size=embedding_dim,
                                  hidden_size=model_size,
                                  num_layers=num_encoder_layers,
                                  batch_first=True)
        self.decoder_rnn = nn.GRU(input_size=embedding_dim + model_size,
                                  hidden_size=model_size,
                                  num_layers=num_decoder_layers,
                                  batch_first=True)

        self.mlp = MLP(hidden_dim=model_size, feedforward_dim=feedforward_dim, output_size=len(ipa_vocab))
        self.attention = Attention(hidden_dim=model_size, embedding_dim=embedding_dim)

    def forward(self, source_tokens, source_langs, target_tokens):
        batch_size = source_tokens.size(0)
        device = source_tokens.device

        # encoder
        # encoder_states: batch size x L x H
        # memory: num_encoder_layers x batch size x H, where L = len(concatenated cognate set)
        (encoder_states, memory), embedded_cognateset = self.encode(source_tokens, source_langs)
        # take the last layer from the RNN - becomes (1, batch_size, H)
        memory = memory[-1, :, :].unsqueeze(dim=0)  # TODO: why isn't batch size first?

        # perform dropout on the output of the RNN
        encoder_states = self.dropout(encoder_states)

        # decoder
        # start of protoform sequence
        start_char = (BOS_IDX * torch.ones((batch_size, 1), dtype=torch.int64).to(device),
                      self.lang_vocab.get_idx(SEPARATOR_LANG) * torch.ones((batch_size, 1), dtype=torch.int64).to(device))
        start_encoded = self.embeddings(*start_char)
        # initialize weighted states to the final encoder state
        attention_weighted_states = memory.transpose(0, 1)
        # start_encoded: batch_size x 1 x E, attention_weighted_states: batch_size x 1 x H; both have seq len of 1
        # concatenated into batch_size x 1 x (H + E)
        decoder_input = torch.cat((start_encoded, attention_weighted_states), dim=-1)
        # perform dropout on the input to the RNN
        decoder_input = self.dropout(decoder_input)
        decoder_state, hidden = self.decoder_rnn(decoder_input)
        # perform dropout on the output of the RNN
        decoder_state = self.dropout(decoder_state)
        hidden = self.dropout(hidden)
        scores = []

        # TODO: verify that there's no one off issue with the time steps (prediction at time step t uses t - 1)

        # target_tokens: (batch size, len of target)
        # individually go through each token, so transpose
        for target_tkn in target_tokens.T:
            target_tkn = target_tkn.unsqueeze(-1)
            # (after) target_tkn: batch_size x 1

            lang = self.protolang if target_tkn not in SPECIAL_TOKENS else SEPARATOR_LANG
            lang = self.lang_vocab.get_idx(lang) * torch.ones((batch_size, 1), dtype=torch.int64).to(device)
            # embedding layer
            # true_char_embedded: (batch_size, 1, E)
            true_char_embedded = self.embeddings(target_tkn, lang)
            # MLP to get a probability distribution over the possible output phonemes
            # char_scores: [batch_size, 1, |Y|] where |Y| = len(self.ipa_vocab)
            char_scores = self.mlp(decoder_state + attention_weighted_states)
            scores.append(char_scores)

            # attention over the encoder states - results in (batch_size, H)
            # decoder_state = query
            # encoder_states = keys
            # encoded_input = embedded_cognateset
            # TODO: what if the encoder hidden dimension != decoder hidden dimension? linear projection to get dims to match in attention?
            # attention_weighted_states: (batch_size, 1, H)
            attention_weighted_states = self.attention(decoder_state, encoder_states, embedded_cognateset)
            attention_weighted_states = attention_weighted_states.unsqueeze(dim=1)
            # decoder_input: (batch_size, 1, H + E)
            # concatenate the attention and the character embedding
            # teacher forcing because we feed in the true target phoneme and don't even take the model's prediction
            decoder_input = torch.cat((true_char_embedded, attention_weighted_states), dim=-1)

            # TODO: check if this still implements variational dropout if num_layers > 1 - may not be true
            # perform dropout on the input to the RNN
            decoder_input = self.dropout(decoder_input)
            # decoder_state: (batch_size, 1, H)
            decoder_state, hidden = self.decoder_rnn(decoder_input, hidden)
            hidden = self.dropout(hidden)
            # perform dropout on the output of the RNN
            decoder_state = self.dropout(decoder_state)

        # |T| elem list with elements of dimension (batch_size, 1, |Y|) -> (batch_size, T, |Y|)
        scores = torch.cat(scores, dim=1)
        return scores

    def encode(self, source_tokens, source_langs):
        # source_tokens and source_langs: (batch size, seq len)
        embedded_cognateset = self.embeddings(source_tokens, source_langs)  # TODO: .to(device)
        # embedded_cognateset: (batch size, seq len, embedding dim)

        # perform dropout on the input to the RNN
        embedded_cognateset = self.dropout(embedded_cognateset)
        return self.encoder_rnn(embedded_cognateset), embedded_cognateset

    def greedy_decode(self, source_tokens, source_langs, max_length):
        batch_size = source_tokens.size(0)
        device = source_tokens.device

        # greedy decoding - generate protoform by picking most likely sequence at each time step
        (encoder_states, memory), embedded_cognateset = self.encode(source_tokens, source_langs)
        # take the last layer from the RNN - becomes (1, batch_size, H)
        memory = memory[-1, :, :].unsqueeze(dim=0)

        start_char = (BOS_IDX * torch.ones((batch_size, 1), dtype=torch.int64).to(device),
                      self.lang_vocab.get_idx(SEPARATOR_LANG) * torch.ones((batch_size, 1), dtype=torch.int64).to(device))
        start_encoded = self.embeddings(*start_char)  # TODO: .to(device)

        # initialize weighted states to the final encoder state
        attention_weighted_states = memory.transpose(0, 1)
        # start_encoded: batch_size x 1 x E, attention_weighted_states: batch_size x 1 x H; both have seq len of 1
        # concatenated into batch_size x 1 x (H + E)
        decoder_input = torch.cat((start_encoded, attention_weighted_states), dim=-1)
        #       perform dropout on the input to the RNN
        # TODO: decoder_input = self.dropout(decoder_input)
        decoder_state, hidden = self.decoder_rnn(decoder_input)
        # TODO: dropout?
        reconstruction = []

        i = 0
        while i < max_length:
            # embedding layer
            # MLP to get a probability distribution over the possible output phonemes
            # char_scores: [batch_size, 1, |Y|] where |Y| = len(self.ipa_vocab)
            char_scores = self.mlp(decoder_state + attention_weighted_states)
            predicted_char = torch.argmax(char_scores.squeeze(dim=1), dim=-1)  #  TODO: .to(device)
            if len(predicted_char) == 1:
                # in the case that batch size is 1
                predicted_char = predicted_char.unsqueeze(dim=0)
            # predicted_char: batch_size x 1
            lang = self.lang_vocab.get_idx(self.protolang) * torch.ones((batch_size, 1), dtype=torch.int64).to(device)
            predicted_char_embedded = self.embeddings(predicted_char, lang)

            # dot product attention over the encoder states
            # attention_weighted_states: (batch_size, 1, H)
            attention_weighted_states = self.attention(decoder_state, encoder_states, embedded_cognateset)
            attention_weighted_states = attention_weighted_states.unsqueeze(dim=1)
            # (batch_size, 1, H + E)
            decoder_input = torch.cat((predicted_char_embedded, attention_weighted_states), dim=-1)

            # TODO: why is there no dropout here? it's because it's only called during evaluation, right?
            # decoder_state: (batch_size, 1, H)
            decoder_state, hidden = self.decoder_rnn(decoder_input, hidden)

            # TODO: check that the prediction for time step t uses t - 1
            reconstruction.append(predicted_char)

            # TODO: check if we need to include the language embedding for the final >?

            i += 1
            # TODO: wait until ALL of them generate EOS_IDX
            # OR just loop MAX_LENGTH times and ensure that all of the tokens after the first EOS are discarded during edit distance calculation
            # end of sequence generated
            if predicted_char.squeeze(-1) == EOS_IDX:
                break

        return torch.tensor(reconstruction)
