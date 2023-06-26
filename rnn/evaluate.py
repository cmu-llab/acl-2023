import os
import torch
import torch.nn as nn
import argparse
from model.encoder_decoder_rnn import EncoderDecoderRNN
from data.dataset import DatasetConcat
from torch.utils.data import DataLoader
import copy
import lingrex.reconstruct
import panphon.distance
from utils import get_edit_distance
from data.special_tokens import *
from train_encoder_decoder_rnn import write_preds
from pathlib import Path
import numpy as np


def evaluate(model, loss_fn, loader, ipa_vocab, dataset_name, print_output=False):
    model.eval()
    total_loss = 0
    char_edit_distance, phoneme_edit_distance = 0, 0
    ned_list, nped_list = [], []
    n_correct = 0
    total_target_phoneme_len = 0
    total_target_len = 0

    predicted_strs, target_strs = [], []
    phoneme_pairs = []

    for i, (source_tokens, dialect, _, target_tokens) in enumerate(loader):
        source_tokens, dialect, target_tokens = source_tokens.to(DEVICE), dialect.to(DEVICE), target_tokens.to(DEVICE)

        # forward computation to get loss
        model_output = model(source_tokens, dialect, target_tokens)
        # outputs (logits): (batch_size, T, |Y|) -> (batch_size, |Y|, T) as required by CrossEntropyLoss
        # target_tokens: (batch_size, T)
        loss = loss_fn(model_output.transpose(1, 2), target_tokens)
        total_loss += loss.item()

        # decoding
        MAX_LENGTH = 30 if 'romance' in args.dataset else 15
        prediction = model.greedy_decode(source_tokens, dialect, MAX_LENGTH)

        target_tkn_list, predicted_tkn_list = ipa_vocab.to_tokens(target_tokens[0]), ipa_vocab.to_tokens(prediction)
        target_str, predicted_str = ''.join(target_tkn_list), ''.join(predicted_tkn_list)
        # non-IPA g gets ignored by panphon when calculating FER
        if "orto" not in dataset_name:
            target_str, predicted_str = target_str.replace('g', 'ɡ'), predicted_str.replace('g', 'ɡ')

        # compute edit distance
        ced = get_edit_distance(target_str, predicted_str)              # character edit distance
        ped = get_edit_distance(target_tkn_list, predicted_tkn_list)    # phoneme (token) edit distance
        char_edit_distance += ced
        phoneme_edit_distance += ped
        ned_list.append(ced / len(target_str))
        nped_list.append(ped / len(target_tkn_list))

        target_strs.append(target_str)
        predicted_strs.append(predicted_str)
        phoneme_pairs.append([target_tkn_list, predicted_tkn_list])

        total_target_phoneme_len += len(target_tkn_list)
        total_target_len += len(target_str)
        if target_str == predicted_str:
            n_correct += 1

        # print sample decoded outputs
        if i < 10 and print_output:
            print(target_str + '  |  ' + predicted_str)

    N = len(loader.dataset)

    # compute FER, BCubed F-score
    dist = panphon.distance.Distance()
    bcubed_f_score = lingrex.reconstruct.eval_by_bcubes(phoneme_pairs)

    test_dict = {
        'test/loss': total_loss / N,
        'test/accuracy': n_correct / N,
        'test/bcubed_f_score': bcubed_f_score,
        'test/target_prediction_pairs': phoneme_pairs,
    }
    if "orto" in dataset_name:
        # for datasets with orthographic transcriptions, not IPA
        # CED - mean across all examples
        test_dict['test/edit_distance'] = char_edit_distance / N
        # NED - CED / length of target, averaged across all examples
        assert len(ned_list) == N
        test_dict['test/normalized_edit_distance'] = sum(ned_list) / N
    else:
        # for datasets in IPA
        # PED - mean across all examples
        test_dict['test/phoneme_edit_distance'] = phoneme_edit_distance / N
        # NPED - PED / length of target, averaged across all examples
        assert len(nped_list) == N
        test_dict['test/normalized_phoneme_edit_distance'] = sum(nped_list) / N
        # PER - sum of all phoneme edits, normalized by total number of target phonemes
        test_dict['test/phoneme_error_rate'] = phoneme_edit_distance / total_target_phoneme_len
        # FER - panphon
        test_dict['test/feature_error_rate'] = dist.feature_error_rate(predicted_strs, target_strs)

    return test_dict


def evaluate_final_checkpoint(model_path, save_preds=False, preds_path='', do_print=False):
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # test on best checkpoints
    saved_info = torch.load(model_path, map_location=torch.device(DEVICE))

    args = saved_info['args']
    # the underlying vocab dicts should be the same as during training
    ipa_vocab = saved_info['ipa_vocab']
    dialect_vocab = saved_info['dialect_vocab']

    model = EncoderDecoderRNN(
        ipa_vocab,
        dialect_vocab,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        feedforward_dim=args.dim_feedforward,
        embedding_dim=args.embedding_size,
        model_size=args.hidden_size,
    ).to(DEVICE)
    model.load_state_dict(saved_info['model'])
    print(sum(p.numel() for p in model.parameters()), 'parameters')

    test_loader = get_dataloader('test', args, ipa_vocab, dialect_vocab)
    test_loss_dict = evaluate(model, loss_fn, test_loader, ipa_vocab, args.dataset, print_output=True)

    if do_print:
        print(model_path)
        print(f'===== <FINAL - best edit_distance>  (epoch: {saved_info["epoch"]}) ======')
        for k, v in test_loss_dict.items():
            if k != 'test/target_prediction_pairs':
                print(k, ' '*(26-len(k)), v)

    if save_preds:
        if 'ipa' in args.dataset:
            evaluate_sound_correspondences(model, ipa_vocab, dialect_vocab, preds_path, args)
        write_preds(preds_path, test_loss_dict['test/target_prediction_pairs'])

    return test_loss_dict


def get_dataloader(partition, args, ipa_vocab, dialect_vocab):
    assert partition in {'train', 'dev', 'test'}, f'invalid partition: {partition}'

    Dataset = DatasetConcat  # Meloni concatenates all daughters

    # load datasets
    fname = f'{partition}.pickle'
    dataset = Dataset(
        fpath=os.path.join(DATA_DIR, args.dataset, fname),
        ipa_vocab=ipa_vocab,
        dialect_vocab=dialect_vocab,
        data_shape=args.data_shape,
        skip_daughter_tone=False,
        skip_protoform_tone=False,
        lang_separators=True,
    )

    print(f'# samples ({partition}): {len(dataset)}')

    if partition == 'train':
        loader_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    else:
        loader_kwargs = {'batch_size': 1, 'shuffle': False}

    dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, pin_memory=True,
                            num_workers=0, **loader_kwargs)
    return dataloader


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
    # load environment variables
    WORK_DIR = os.environ.get('WORK_DIR')
    DATA_DIR = os.environ.get('DATA_DIR')
    # evaluation code does not work on MPS
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()
    # parser.add_argument('--sweep', type=str, required=True, help='wandb sweep name')
    parser.add_argument('--model', type=str, default='rnn', help='rnn')
    parser.add_argument('--dataset', type=str, default='chinese_baxter',
                        help='chinese_wikihan2022/chinese_baxter/romance_ipa/romance_orto')

    args = parser.parse_args()

    predictions_dir = os.path.join(WORK_DIR, 'predictions', f'{args.model}_{args.dataset}')
    Path(predictions_dir).mkdir(parents=True, exist_ok=True)
    preds_path = os.path.join(predictions_dir, 'best_ed.txt')

    # evaluate 10 runs
    ped, nped, ced, nced, accuracy, fer, bcfs = [], [], [], [], [], [], []
    runs_dir = os.path.join(WORK_DIR, 'checkpoints', f'{args.model}_{args.dataset}')
    runs = Path(runs_dir).rglob(f'{args.model}_{args.dataset}_gpu*_best_ed.pt')

    for model_path in runs:
        # only save predictions on the best run (the run with the best dev ED)
        if "best" in str(model_path.parent):
            metrics = evaluate_final_checkpoint(model_path, save_preds=True, preds_path=preds_path, do_print=True)
        else:
            metrics = evaluate_final_checkpoint(model_path)
        accuracy.append(metrics['test/accuracy'])
        bcfs.append(metrics['test/bcubed_f_score'])
        if "orto" in args.dataset:
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
    if "orto" in args.dataset:
        print('ED', ced.mean(), ced.std())
        print('NED', nced.mean(), nced.std())
    else:
        print('PED', ped.mean(), ped.std())
        print('NPED', nped.mean(), nped.std())
        print('FER', fer.mean(), fer.std())
    print('Accuracy', accuracy.mean(), accuracy.std())
    print('BCFS', bcfs.mean(), bcfs.std())
