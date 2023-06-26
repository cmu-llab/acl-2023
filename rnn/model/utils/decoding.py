import torch
import torch.nn as nn

from data.special_tokens import *
from utils import reparameterize


def greedy_decode(model, source, dialect, source_mask, source_padding_mask, max_len, device):
    memory = model.encode(source, dialect, source_mask, source_padding_mask)
    if isinstance(memory, tuple):
        memory = memory[0]
    generated_sequences = []
    for source_memory in memory:
        ys = torch.ones(1, 1).fill_(BOS_IDX).long().to(device)
        for i in range(max_len - 1):
            target_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1)).bool().to(device)
            out = model.decode_proto(ys, torch.unsqueeze(source_memory, 0), target_mask)
            prob = model._generator(out[:, -1])
            _, next_token_idx = torch.max(prob, dim=1)
            next_token_idx = next_token_idx.item()

            ys = torch.cat((ys, torch.ones(1, 1).fill_(next_token_idx).long().to(device)), dim=1)
            if next_token_idx == EOS_IDX:
                break
        generated_sequences.append(torch.squeeze(ys))
    return generated_sequences


def greedy_decode_ensemble(models, source, dialect, source_mask, source_padding_mask, max_len, device):
    # encode in batch
    memories = [model.encode(source, dialect, source_mask, source_padding_mask) for model in models]
    if isinstance(memories[0], tuple):
        memories = [memory[0] for memory in memories]
    memories = torch.stack(memories)  # n_checkpoints x batch x source_seq_len x d_model
    memories = memories.transpose(0, 1)  # batch x n_checkpoints x source_seq_len x d_model

    # decode one sample at a time
    generated_sequences = []
    for source_memories in memories:
        ys = torch.ones(1, 1).fill_(BOS_IDX).long().to(device)
        for i in range(max_len - 1):
            # generate output probabilities for each checkpoint
            probs = []
            for model, source_memory in zip(models, source_memories):
                target_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1)).bool().to(device)
                out = model.decode_proto(ys, torch.unsqueeze(source_memory, 0), target_mask)
                prob = model._generator(out[:, -1])
                probs.append(prob)

            # ensemble
            ensembled_prob = torch.mean(torch.stack(probs), dim=0)

            _, next_token_idx = torch.max(ensembled_prob, dim=1)
            next_token_idx = next_token_idx.item()

            ys = torch.cat((ys, torch.ones(1, 1).fill_(next_token_idx).long().to(device)), dim=1)
            if next_token_idx == EOS_IDX:
                break
        generated_sequences.append(torch.squeeze(ys))
    return generated_sequences


def greedy_decode_daughter(model, source, dialect, source_mask, source_padding_mask, max_len, sampled_daughters, device):
    memory, memory_attention, source_embedding = model.encode(source, dialect, source_mask, source_padding_mask)
    mu = model._to_mu(memory_attention)
    logvar = model._to_logvar(memory_attention)
    z = reparameterize(mu, logvar)  # [B, E]
    sampled_dialects = dialect[[range(dialect.size(0)), sampled_daughters]]

    generated_sequences = []
    for a_memory, a_z, a_source_emb, a_dialect in zip(memory, z, source_embedding, sampled_dialects):
        # onedtr_embedding, onedtr_mask, onedtr_padding_mask, sampled_daughters \
        #     = model.sample_one_daughter(a_source_emb.unsqueeze(0), source_padding_mask)
        ys = torch.ones(1, 1).fill_(BOS_IDX).long().to(device)
        for i in range(max_len - 1):
            target_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1)).bool().to(device)
            onedtr_embedding = model._ipa_embedding(ys.unsqueeze(1)).squeeze(1) + model._dialect_embedding(a_dialect).unsqueeze(0)  # [B, 1, T, E]
            out, _ = model.decode_daughter(onedtr_embedding, target_mask,
                                        daughter_hidden=a_z.unsqueeze(0))
            prob = model._daughter_generator(out[:, -1])
            _, next_token_idx = torch.max(prob, dim=1)
            next_token_idx = next_token_idx.item()

            ys = torch.cat((ys, torch.ones(1, 1).fill_(next_token_idx).long().to(device)), dim=1)
            if next_token_idx == EOS_IDX:
                break
        generated_sequences.append(torch.squeeze(ys))
    return generated_sequences