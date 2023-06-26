import argparse

from torch.nn import CrossEntropyLoss

from data.special_tokens import *
import random
import torch
import numpy as np


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_kl_loss(mu, logvar):
    kl_loss = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()
    return kl_loss


def reparameterize(mu, logvar, sample=True):
    if sample:
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)
    else:
        return mu


def calc_loss(model_output, target, source, loss_multipliers):
    cross_entropy = CrossEntropyLoss(ignore_index=PAD_IDX)

    target_out = target[:, 1:]  # shift back
    if isinstance(model_output, tuple):
        proto_logits, daughter_logits, mu, logvar, sampled_daughters = model_output
        source_out = source[range(source.shape[0]), sampled_daughters, 1:]  # shift back
    else:
        proto_logits = model_output
        daughter_logits, mu, logvar = None, None, None

    proto_recon_loss = cross_entropy(proto_logits.reshape((-1, proto_logits.shape[-1])),
                         target_out.reshape(-1)) * loss_multipliers['proto_recon']

    dtr_recon_loss, kl_loss = 0, 0
    if daughter_logits is not None:
        dtr_recon_loss = cross_entropy(daughter_logits.reshape((-1, daughter_logits.shape[-1])),
                         source_out.reshape(-1)) * loss_multipliers['daughter_recon']
        kl_loss = get_kl_loss(mu, logvar) * loss_multipliers['kl']

    return proto_recon_loss, dtr_recon_loss, kl_loss


def get_edit_distance(s1, s2):
    # source: https://github.com/shauli-ravfogel/Latin_reconstruction
    if type(s1) == str and type(s2) == str:
        s1 = s1.replace("<", "").replace(">", "")
        s2 = s2.replace("<", "").replace(">", "")

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
