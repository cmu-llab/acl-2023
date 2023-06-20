import os
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import copy
import lingrex.reconstruct
import panphon.distance
from pathlib import Path
import numpy as np

from model import Model
from data import Dataset
from special_tokens import *
from utils import load_config, get_edit_distance, dict_to_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, required=True)

    args = parser.parse_args()

    return args


def load_environment_variables():
    global WORK_DIR
    global SRC_DIR
    global DATA_DIR
    global CONF_DIR

    WORK_DIR = os.environ['WORK_DIR']
    SRC_DIR = os.environ['SRC_DIR']
    DATA_DIR = os.environ['DATA_DIR']
    CONF_DIR = os.environ['CONF_DIR']


def declare_globals(conf):
    global DEVICE
    global MAX_LENGTH
    global DATASET

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MAX_LENGTH = 30 if 'romance' in conf.dataset else 15
    DATASET = conf.dataset


def get_dataset_path(split):
    return os.path.join(DATA_DIR, DATASET, f'{split}.pickle')


def initialize_model(conf, ipa_vocab, dialect_vocab):
    model = Model(
        num_encoder_layers=conf.num_encoder_layers,
        num_decoder_layers=conf.num_decoder_layers,
        embedding_size=conf.embedding_size,
        nhead=conf.nhead,
        dim_feedforward=conf.dim_feedforward,
        dropout=conf.dropout,
        max_length=MAX_LENGTH,
        ipa_vocab=ipa_vocab,
        dialect_vocab=dialect_vocab
    ).to(DEVICE)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


def create_mask(src, tgt):
    '''
    * src: batch of source index sequences (batch x seq_len)
    * tgt: batch of target index sequences (batch x seq_len)
    '''
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).bool().to(DEVICE)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).bool().to(DEVICE)

    src_padding_mask = (src == PAD_IDX)
    tgt_padding_mask = (tgt == PAD_IDX)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def evaluate(model, criterion, loader, ipa_vocab):
    model.eval()

    total_loss = 0
    char_edit_distance, phoneme_edit_distance = 0, 0
    ned_list, nped_list = [], []
    n_correct, n_total = 0, 0
    total_target_phoneme_len = 0

    ret_predictions = []
    predicted_strs, target_strs = [], []
    for source, dialect, source_lens, target in loader:
        source, dialect, source_lens, target = \
            source.to(DEVICE), dialect.to(DEVICE), source_lens.to(DEVICE), target.to(DEVICE)
        target_in = target[:, :-1]
        source_mask, target_mask, source_padding_mask, target_padding_mask = create_mask(source, target_in)

        logits = model(
            source,
            dialect,
            source_lens,
            target_in,
            source_mask,
            target_mask,
            source_padding_mask,
            target_padding_mask,
            source_padding_mask
        )

        target_out = target[:, 1:]
        loss = criterion(logits.reshape((-1, logits.shape[-1])), target_out.reshape(-1))
        total_loss += loss.item()

        predictions = greedy_decode(model, source, source_lens, dialect, source_mask, source_padding_mask, MAX_LENGTH)

        for gold_seq, prediction in zip(target, predictions):
            gold_char_list = ipa_vocab.to_string(gold_seq, return_list=True)
            pred_char_list = ipa_vocab.to_string(prediction, return_list=True)

            ret_predictions.append((gold_char_list, pred_char_list))
            gold_char_str, pred_char_str = ''.join(gold_char_list), ''.join(pred_char_list)
            # non-IPA g gets ignored by panphon when calculating FER
            if "orto" not in DATASET:
                gold_char_str, pred_char_str = gold_char_str.replace('g', 'ɡ'), pred_char_str.replace('g', 'ɡ')
            predicted_strs.append(pred_char_str)
            target_strs.append(gold_char_str)

            ped = get_edit_distance(gold_char_list, pred_char_list)
            ced = get_edit_distance(gold_char_str, pred_char_str)
            phoneme_edit_distance += ped
            char_edit_distance += ced
            total_target_phoneme_len += len(gold_char_list)
            ned_list.append(ced / len(gold_char_str))
            nped_list.append(ped / len(gold_char_list))

            if gold_char_str == pred_char_str:
                n_correct += 1
            n_total += 1

    # compute FER, BCubed F-score
    dist = panphon.distance.Distance()
    bcubed_f_score = lingrex.reconstruct.eval_by_bcubes(ret_predictions)

    test_dict = {
        'test/loss': total_loss / n_total,
        'test/accuracy': n_correct / n_total,
        'test/bcubed_f_score': bcubed_f_score,
        'test/target_prediction_pairs': ret_predictions,
    }
    if "orto" in DATASET:
        # for datasets with orthographic transcriptions, not IPA
        # CED - mean across all examples
        test_dict['test/edit_distance'] = char_edit_distance / n_total
        # NED - CED / length of target, averaged across all examples
        assert len(ned_list) == n_total
        test_dict['test/normalized_edit_distance'] = sum(ned_list) / n_total
    else:
        # for datasets in IPA
        # PED - mean across all examples
        test_dict['test/phoneme_edit_distance'] = phoneme_edit_distance / n_total
        # NPED - PED / length of target, averaged across all examples
        assert len(nped_list) == n_total
        test_dict['test/normalized_phoneme_edit_distance'] = sum(nped_list) / n_total
        # PER - sum of all phoneme edits, normalized by total number of target phonemes
        test_dict['test/phoneme_error_rate'] = phoneme_edit_distance / total_target_phoneme_len
        # FER - panphon
        test_dict['test/feature_error_rate'] = dist.feature_error_rate(predicted_strs, target_strs)

    return test_dict


def load_model(filepath):
    saved_info = torch.load(filepath, map_location=torch.device(DEVICE))
    return saved_info


def test(criterion, model_path, save_preds=False, preds_path='', do_print=False):
    saved_info = load_model(model_path)

    conf = dict_to_config(saved_info['conf'])
    ipa_vocab = saved_info['ipa_vocab']
    dialect_vocab = saved_info['dialect_vocab']

    model = Model(
        num_encoder_layers=conf.num_encoder_layers,
        num_decoder_layers=conf.num_decoder_layers,
        embedding_size=conf.embedding_size,
        nhead=conf.nhead,
        dim_feedforward=conf.dim_feedforward,
        dropout=conf.dropout,
        max_length=MAX_LENGTH,
        ipa_vocab=ipa_vocab,
        dialect_vocab=dialect_vocab
    ).to(DEVICE)

    model.load_state_dict(saved_info['model'])

    print(sum(p.numel() for p in model.parameters()), 'parameters')

    test_dataset = Dataset(get_dataset_path('test'), ipa_vocab, dialect_vocab)
    test_loader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False,
                             collate_fn=test_dataset.collate_fn)

    metrics = evaluate(model, criterion, test_loader, ipa_vocab)

    if do_print:
        print(f'===== <FINAL - best edit_distance>  (epoch: {saved_info["epoch"]}) ======')
        for k, v in metrics.items():
            if k != 'test/target_prediction_pairs':
                print(k, ' ' * (26 - len(k)), v)
        print()
    if save_preds:
        # TODO:
        # if 'ipa' in DATASET:
        #     evaluate_sound_correspondences(model, ipa_vocab, dialect_vocab, preds_path, args)
        write_preds(preds_path, metrics['test/target_prediction_pairs'])

    return metrics


def write_preds(filepath, target_prediction_pairs):
    # predictions: predicted - target protoform (both will be phoneme tokenized
    with open(filepath, 'w') as f:
        f.write("gold standard\tprediction\n")
        for pair in target_prediction_pairs:
            gold_std, pred = pair
            f.write(f"{' '.join(gold_std)}\t{' '.join(pred)}\n")

    print(f'\t>> saved predictions to {filepath}')


def greedy_decode(model, source, source_lens, dialects, source_mask, source_padding_mask, max_len):
    memory = model.encode(source, source_lens, dialects, source_mask, source_padding_mask)
    generated_sequences = []
    for source_memory in memory:
        latent_ys = torch.ones(1, 1).fill_(BOS_IDX).long().to(DEVICE)
        for i in range(max_len - 1):
            target_mask = nn.Transformer.generate_square_subsequent_mask(latent_ys.size(1)).bool().to(DEVICE)
            out = model.decode(latent_ys, torch.unsqueeze(source_memory, axis=0), target_mask)
            prob = model._generator(out[:, -1])
            _, latent_next_idx = torch.max(prob, dim=1)
            latent_next_idx = latent_next_idx.item()

            latent_ys = torch.cat((latent_ys, torch.ones(1, 1).fill_(latent_next_idx).long().to(DEVICE)), dim=1)
            if latent_next_idx == EOS_IDX:
                break
        generated_sequence = torch.squeeze(latent_ys)
        generated_sequences.append(generated_sequence)
    return generated_sequences


def evaluate_sound_correspondences(model, ipa_vocab, dialect_vocab, preds_path, args):
    args = copy.deepcopy(args)
    args.dataset = 'romance_sound_correspondences'
    sound_correspondence_loader = get_dataloader('test', args, ipa_vocab, dialect_vocab)
    model.eval()
    n_correct = 0
    new_path = os.path.join(Path(preds_path).parent, 'sound_correspondences.tsv')
    phoneme_pairs = []

    for i, (source_tokens, dialect, _, target_tokens) in enumerate(sound_correspondence_loader):
        # EXP
        source_tokens, dialect, target_tokens = source_tokens.to(DEVICE), dialect.to(DEVICE), target_tokens.to(DEVICE)

        # decoding
        prediction = model.greedy_decode(source_tokens, dialect, 30 if 'romance' in args.dataset else 15)

        target_tkn_list, predicted_tkn_list = ipa_vocab.to_tokens(target_tokens[0]), ipa_vocab.to_tokens(prediction)
        target_str, predicted_str = ''.join(target_tkn_list), ''.join(predicted_tkn_list)
        target_str, predicted_str = target_str.replace('g', 'ɡ'), predicted_str.replace('g', 'ɡ')

        phoneme_pairs.append([target_tkn_list, predicted_tkn_list])

        if target_str == predicted_str:
            n_correct += 1

    write_preds(new_path, phoneme_pairs)
    N = len(sound_correspondence_loader.dataset)
    print("Accuracy of sound correspondences", n_correct / N)


if __name__ == '__main__':
    args = parse_args()
    load_environment_variables()
    conf = load_config(os.path.join(CONF_DIR, f'{args.conf}.json'))
    declare_globals(conf)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    predictions_dir = os.path.join(WORK_DIR, 'predictions', f'transformer_{conf.dataset}')
    Path(predictions_dir).mkdir(parents=True, exist_ok=True)
    preds_path = os.path.join(predictions_dir, 'best_ed.txt')

    # evaluate 5 runs
    ped, nped, ced, nced, accuracy, fer, bcfs = [], [], [], [], [], [], []
    runs_dir = os.path.join(WORK_DIR, 'checkpoints', f'transformer_{conf.dataset}')
    runs = Path(runs_dir).rglob(f'transformer_{conf.dataset}_gpu*_best_ed.pt')

    for model_path in runs:
        # only save predictions on the best run (the run with the best dev ED)
        if "best" in str(model_path.parent):
            metrics = test(criterion, model_path, save_preds=True, preds_path=preds_path, do_print=True)
        else:
            metrics = test(criterion, model_path)
        accuracy.append(metrics['test/accuracy'])
        bcfs.append(metrics['test/bcubed_f_score'])
        if "orto" in conf.dataset:
            ced.append(metrics['test/edit_distance'])
            nced.append(metrics['test/normalized_edit_distance'])
        else:
            ped.append(metrics['test/phoneme_edit_distance'])
            nped.append(metrics['test/normalized_phoneme_edit_distance'])
            fer.append(metrics['test/feature_error_rate'])
    ped, nped, ced, nced, accuracy, fer, bcfs = np.array(ped), np.array(nped), np.array(ced), np.array(nced), \
                                                np.array(accuracy), np.array(fer), np.array(bcfs)
    print('\nmean / stdev across 10 runs')
    assert len(accuracy) == 10
    if "orto" in conf.dataset:
        print('ED', ced.mean(), ced.std())
        print('NED', nced.mean(), nced.std())
    else:
        print('PED', ped.mean(), ped.std())
        print('NPED', nped.mean(), nped.std())
        print('FER', fer.mean(), fer.std())
    print('Accuracy', accuracy.mean(), accuracy.std())
    print('BCFS', bcfs.mean(), bcfs.std())
