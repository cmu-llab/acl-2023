import os
import time
import logging
import argparse

import wandb
import transformers
import panphon.distance
import lingrex.reconstruct

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import DatasetConcat
from data.utils import build_vocab
from model.encoder_decoder_rnn import EncoderDecoderRNN
from utils import str2bool, get_edit_distance, seed_everything
from data.special_tokens import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='chinese_wikihan2022')
    parser.add_argument('--data_shape', type=str, choices=['stack', 'concatenate'], default='concatenate')
    parser.add_argument('--dump_results', type=str2bool, default=False, help='whether or not to record model results')
    parser.add_argument('--lr', type=float, default=0.0015, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 in Adam')
    parser.add_argument('--beta2', type=float, default=0.98, help='beta2 in Adam')
    parser.add_argument('--encoder_layers', type=int, default=3, help='number of encoder layers')  # TODO: set this to 1 when doing sanity check
    parser.add_argument('--decoder_layers', type=int, default=3, help='number of decoder layers')  # TODO: set this to 1 when doing sanity check
    parser.add_argument('--embedding_size', type=int, default=128, help='embedding dimension')
    parser.add_argument('--dim_feedforward', type=int, default=128, help='dimension of feedforward network in the decoder')
    parser.add_argument('--dropout', type=float, default=0.07)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lang_separators', type=str2bool, default=True,
                        help='whether or not to add language separator tokens between each cognate when concatenated')
    parser.add_argument('--warmup_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=1)  # TODO: don't batch yet
    parser.add_argument('--wandb_name', type=str, default="")
    parser.add_argument('--wandb_entity', type=str, default="llab-reconstruction")
    parser.add_argument('--sweeping', type=str2bool, default=False)
    parser.add_argument('--save_predictions', type=str2bool, default=False)

    # RNN-specific
    parser.add_argument('--hidden_size', type=int, default=50, help='RNN hidden layer size')

    parser.add_argument('--seed', type=int, default=12345, help='seed')
    
    return parser.parse_args()


def initialize_model(args, ipa_vocab, dialect_vocab):
    model = EncoderDecoderRNN(
        ipa_vocab,
        dialect_vocab,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        feedforward_dim=args.dim_feedforward,
        embedding_dim=args.embedding_size,
        model_size=args.hidden_size,
    )

    # Meloni et al do not do Xavier initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model.to(DEVICE)


def get_dataloader(partition, args):
    assert partition in {'train', 'dev', 'test'}, f'invalid partition: {partition}'

    Dataset = DatasetConcat  # Meloni concatenates all daughters
    # TODO: stack Meloni
    
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
        # TODO: enable batching during decoding
        loader_kwargs = {'batch_size': 1, 'shuffle': False}

    dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, pin_memory=True,
        num_workers=0, **loader_kwargs)
    return dataloader


def train_once(model, optimizer, loss_fn, train_loader):
    model.train()

    # Torch already shuffles the order
    total_loss, correct, wrong = 0, 0, 0
    # torch.autograd.set_detect_anomaly(True)
    for i, (source_tokens, dialect, _, target_tokens) in enumerate(train_loader):
        source_tokens, dialect, target_tokens = source_tokens.to(DEVICE), dialect.to(DEVICE), target_tokens.to(DEVICE)
        # TODO: create mask here when we batch

        # forward
        outputs = model(source_tokens, dialect, target_tokens)
        # outputs (logits): (batch_size, T, |Y|) -> (batch_size, |Y|, T) as required by CrossEntropyLoss
        # target_tokens: (batch_size, T)
        loss = loss_fn(outputs.transpose(1, 2), target_tokens)

        # gradient update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predicted = torch.argmax(outputs, dim=-1)
        # predicted: (batch_size, T)
        # note that because this is training and we're using teacher forcing, predicted seq len = target seq len
        # checks if each element is equal
        elementwise_eq = torch.eq(predicted, target_tokens)
        # compare across each batch
        batchwise_accuracy = elementwise_eq.all(dim=-1)

        correct += len(batchwise_accuracy.nonzero())
        wrong += len(batchwise_accuracy == False)

    # since we use sum reduction across batches in the loss, N should be length of the dataset, NOT number of batches
    N = len(train_loader.dataset)
    return {
        'train/loss': total_loss / N,
        'train/accuracy': correct / (correct + wrong)
    }


def dump_results(best_loss_epoch, best_loss, best_ed_epoch, edit_distance, args):
    results_dir = os.path.join(WORK_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # find next exp number
    subdirs = set(os.listdir(results_dir))
    exp_num = 1
    while True:
        exp_path = f'exp_{exp_num}'
        if exp_path in subdirs:
            exp_num += 1
            continue
        exp_dir = os.path.join(results_dir, exp_path)
        os.makedirs(exp_dir)
        break
    
    with open(os.path.join(exp_dir, 'params.txt'), 'w') as fout:
        for k, v in vars(args).items():
            fout.write(f'{k}: {v}\n')
    with open(os.path.join(exp_dir, 'metrics.txt'), 'w') as fout:
        fout.write(f'loss: {best_loss} (epoch {best_loss_epoch})\n')
        fout.write(f'edit_distance: {edit_distance} (epoch {best_ed_epoch})\n')

    print(f'results recorded at {exp_dir}')


def train(model, optimizer, scheduler, loss_fn, ipa_vocab, dialect_vocab, args):
    # load datasets
    train_loader = get_dataloader('train', args)
    dev_loader = get_dataloader('dev', args)
    # TODO: is this any different from re-using the tensors

    # start training
    best_dev_loss = 1e10;best_loss_epoch = -1
    best_edit_distance = 1e10;best_ed_epoch = -1
    for epoch in range(args.epochs):
        # train
        t = time.time()
        train_loss_dict = train_once(model, optimizer, loss_fn, train_loader)

        # validate
        train_time = time.time()
        dev_loss_dict = evaluate(model, loss_fn, dev_loader, print_output=not epoch%10)

        # print results
        print(f'< epoch {epoch} >  (elapsed: {time.time() - t:.2f}s, decode time: {time.time() - train_time:.2f}s)')
        print(f'  * [train]  loss: {train_loss_dict["train/loss"]:.6f}')
        print(f'  * [ dev ]  loss: {dev_loss_dict["dev/loss"]:.6f}  ||  edit distance: {dev_loss_dict["dev/edit_distance"]}  ||  accuracy: {dev_loss_dict["dev/accuracy"]}')
        wandb.log({"train/lr": optimizer.param_groups[0]['lr'], **train_loss_dict, **dev_loss_dict})

        # save checkpoints
        dev_loss = dev_loss_dict['dev/loss']
        # TODO: starting using phoneme edit distance to pick btwn models
        edit_distance = dev_loss_dict['dev/edit_distance']
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_loss_epoch = epoch
            save_model(model, optimizer, args, ipa_vocab, dialect_vocab, epoch, MODELPATH_LOSS)
        if edit_distance < best_edit_distance:
            best_edit_distance = edit_distance
            best_ed_epoch = epoch
            save_model(model, optimizer, args, ipa_vocab, dialect_vocab, epoch, MODELPATH_ED)
        print()
        scheduler.step()

    if args.dump_results:
        dump_results(best_loss_epoch, best_dev_loss, best_ed_epoch, best_edit_distance, args)


@torch.no_grad()
def evaluate(model, loss_fn, loader, print_output=False, eval_name='dev'):
    model.eval()
    total_loss = 0
    char_edit_distance, phoneme_edit_distance = 0, 0
    n_correct = 0
    total_target_phoneme_len = 0
    total_target_len = 0

    predicted_strs, target_strs = [], []
    phoneme_pairs = []

    for i, (source_tokens, dialect, _, target_tokens) in enumerate(loader):
        # EXP
        source_tokens, dialect, target_tokens = source_tokens.to(DEVICE), dialect.to(DEVICE), target_tokens.to(DEVICE)
        # TODO: create padding here when we batch?

        # forward computation to get loss
        model_output = model(source_tokens, dialect, target_tokens)
        # outputs (logits): (batch_size, T, |Y|) -> (batch_size, |Y|, T) as required by CrossEntropyLoss
        # target_tokens: (batch_size, T)
        loss = loss_fn(model_output.transpose(1, 2), target_tokens)
        total_loss += loss.item()

        # decoding
        prediction = model.greedy_decode(source_tokens, dialect, MAX_LENGTH)

        # TODO: batch the predictions here!!
        target_tkn_list, predicted_tkn_list = ipa_vocab.to_tokens(target_tokens[0]), ipa_vocab.to_tokens(prediction)
        target_str, predicted_str = ''.join(target_tkn_list), ''.join(predicted_tkn_list)

        # compute edit distance
        char_edit_distance += get_edit_distance(target_str, predicted_str)  # character edit distance
        phoneme_edit_distance += get_edit_distance(target_tkn_list, predicted_tkn_list)  # phoneme (token) edit distance

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
    feature_error_rate = dist.feature_error_rate(predicted_strs, target_strs)
    bcubed_f_score = lingrex.reconstruct.eval_by_bcubes(phoneme_pairs)

    return {
        f'{eval_name}/loss': total_loss/N,
        f'{eval_name}/edit_distance': char_edit_distance/N,
        f'{eval_name}/normalized_edit_distance': char_edit_distance / total_target_len,
        f'{eval_name}/accuracy': n_correct/N,
        f'{eval_name}/phoneme_edit_distance': phoneme_edit_distance/N,
        f'{eval_name}/phoneme_error_rate': phoneme_edit_distance/total_target_phoneme_len,
        f'{eval_name}/feature_error_rate': feature_error_rate,
        f'{eval_name}/bcubed_f_score': bcubed_f_score,
        f'{eval_name}/target_prediction_pairs': phoneme_pairs,
    }


def evaluate_final_checkpoint(save_predictions=False, predictions_dir=''):
    # test on best checkpoints
    for filepath, criterion in [(MODELPATH_LOSS, 'loss'), (MODELPATH_ED, 'edit_distance')]:
        saved_info = load_model(filepath)

        args = saved_info['args']
        ipa_vocab = saved_info['ipa_vocab']
        dialect_vocab = saved_info['dialect_vocab']

        model = initialize_model(args, ipa_vocab, dialect_vocab)
        model.load_state_dict(saved_info['model'])

        dev_loader = get_dataloader('dev', args)
        test_loader = get_dataloader('test', args)

        dev_loss_dict = evaluate(model, loss_fn, dev_loader, print_output=True)
        test_loss_dict = evaluate(model, loss_fn, test_loader, print_output=True, eval_name='test')

        print(f'===== <FINAL - best {criterion}>  (epoch: {saved_info["epoch"]}) ======')
        for loss_dict in (dev_loss_dict, test_loss_dict):
            for k, v in loss_dict.items():
                print(k, ' '*(26-len(k)), v)

        wandb.run.summary[f'best_dev_loss_by_{criterion}'] = dev_loss_dict['dev/loss']
        wandb.run.summary[f'best_dev_ed_by_{criterion}'] = dev_loss_dict['dev/edit_distance']
        wandb.run.summary[f'best_dev_ned_by_{criterion}'] = dev_loss_dict['dev/normalized_edit_distance']
        wandb.run.summary[f'best_dev_acc_by_{criterion}'] = dev_loss_dict['dev/accuracy']
        wandb.run.summary[f'best_dev_per_by_{criterion}'] = dev_loss_dict['dev/phoneme_error_rate']
        wandb.run.summary[f'best_dev_fer_by_{criterion}'] = dev_loss_dict['dev/feature_error_rate']
        wandb.run.summary[f'best_dev_bcubed_f_by_{criterion}'] = dev_loss_dict['dev/bcubed_f_score']
        wandb.run.summary[f'best_dev_PED_by_{criterion}'] = dev_loss_dict['dev/phoneme_edit_distance']

        wandb.run.summary[f'best_test_loss_by_{criterion}'] = test_loss_dict['test/loss']
        wandb.run.summary[f'best_test_ed_by_{criterion}'] = test_loss_dict['test/edit_distance']
        wandb.run.summary[f'best_test_ned_by_{criterion}'] = test_loss_dict['test/normalized_edit_distance']
        wandb.run.summary[f'best_test_acc_by_{criterion}'] = test_loss_dict['test/accuracy']
        wandb.run.summary[f'best_test_acc_by_{criterion}'] = test_loss_dict['test/accuracy']
        wandb.run.summary[f'best_test_per_by_{criterion}'] = test_loss_dict['test/phoneme_error_rate']
        wandb.run.summary[f'best_test_fer_by_{criterion}'] = test_loss_dict['test/feature_error_rate']
        wandb.run.summary[f'best_test_bcubed_f_by_{criterion}'] = test_loss_dict['test/bcubed_f_score']
        wandb.run.summary[f'best_test_PED_by_{criterion}'] = test_loss_dict['test/phoneme_edit_distance']

        if save_predictions:
            predictions_file = predictions_dir + f'/{MODEL}_{args.dataset}_gpu{GPUID}_best_{criterion}.txt'
            write_preds(predictions_file, test_loss_dict['target_prediction_pairs'])


def save_model(model, optimizer, args, ipa_vocab, dialect_vocab, epoch, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'epoch': epoch,
        'ipa_vocab': ipa_vocab,
        'dialect_vocab': dialect_vocab,
        'wandb_run': wandb.run.name,
    }
    torch.save(save_info, filepath)
    print(f'\t>> saved model to {filepath}')


def load_model(filepath):
    saved_info = torch.load(filepath)
    return saved_info


def write_preds(filepath, target_prediction_pairs):
    print(f'\t>> saved predictions to {filepath}')

    # predictions: predicted - target protoform (both will be phoneme tokenized
    with open(filepath, 'w') as f:
        f.write("gold standard\tprediction\n")
        for pair in target_prediction_pairs:
            gold_std, pred = pair
            f.write(f"{' '.join(gold_std)}\t{' '.join(pred)}\n")


if __name__ == '__main__':
    torch.set_num_threads(5)  # on patient there are 20 CPUs and 4 GPUs, so each job should take max 5 CPUs

    # get commandline arguments
    args = parse_args()
    print(args)
    seed_everything(args.seed)

    MODEL = "rnn"

    # load environment variables
    WORK_DIR = os.environ.get('WORK_DIR')
    DATA_DIR = os.environ.get('DATA_DIR')

    # globals
    DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    GPUID = os.environ.get('CUDA_VISIBLE_DEVICES', 0)
    MAX_LENGTH = 30 if 'romance' in args.dataset else 15

    # Disable lingpy and lingrex redundant logs
    logging.getLogger('lingpy').setLevel(logging.ERROR)

    # initialize wandb
    wandb.init(project="mcr", name=args.wandb_name, entity=args.wandb_entity, dir="../mcr_wandb",
               mode='disabled' if (not args.wandb_name and not args.sweeping) else 'online')
    wandb.run.log_code(".", include_fn=lambda path: path.endswith('train.py') or path.endswith('model.py'))
    wandb.config.update(args)

    # create folders to store best model checkpoints
    os.makedirs(os.path.join(WORK_DIR, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(WORK_DIR, 'checkpoints', wandb.run.name), exist_ok=True)
    MODELPATH_LOSS = os.path.join(WORK_DIR, 'checkpoints', wandb.run.name,
                                  f'{MODEL}_{args.dataset}_gpu{GPUID}_best_loss.pt')
    MODELPATH_ED = os.path.join(WORK_DIR, 'checkpoints', wandb.run.name,
                                f'{MODEL}_{args.dataset}_gpu{GPUID}_best_ed.pt')

    print(f'device: {DEVICE}')
    print(f'gpu id: {GPUID}')

    # build vocabularies from train data
    ipa_vocab, dialect_vocab = build_vocab(os.path.join(DATA_DIR, args.dataset, 'train.pickle'), lang_separators=args.lang_separators)

    # initialize model
    model = initialize_model(args, ipa_vocab, dialect_vocab)

    # prepare training
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=1e-8
    )
    scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
        optimizer,
        args.warmup_epochs,
        args.epochs,
        lr_end=0.000001
    )
    loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=PAD_IDX)

    # train
    train(model, optimizer, scheduler, loss_fn, ipa_vocab, dialect_vocab, args)

    # test
    if args.save_predictions:
        predictions_dir = os.path.join(WORK_DIR, 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)
        evaluate_final_checkpoint(args.save_predictions, predictions_dir)
    else:
        evaluate_final_checkpoint()
