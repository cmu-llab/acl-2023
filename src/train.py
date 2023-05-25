import os
import time
import argparse

import transformers

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import Model
from data import Dataset
from special_tokens import *
from utils import build_vocab, load_config, get_edit_distance, dict_to_config, seed_everything


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


def get_checkpoint_path(metric):
    return os.path.join(WORK_DIR, 'checkpoints', DATASET, f'exp_{EXP_NUM}', f'{metric}.pt')


def get_dataset_path(split):
    return os.path.join(DATA_DIR, DATASET, f'{split}.pickle')


def setup_experiment():
    # checkpoints
    os.makedirs(os.path.join(WORK_DIR, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(WORK_DIR, 'checkpoints', DATASET), exist_ok=True)

    global EXP_NUM
    EXP_NUM = 1
    while True:
        exp_checkpoint_dir = os.path.join(WORK_DIR, 'checkpoints', DATASET, f'exp_{EXP_NUM}')

        if os.path.exists(exp_checkpoint_dir):
            EXP_NUM += 1
        else:
            os.makedirs(exp_checkpoint_dir)
            print(f'exp_num: {EXP_NUM}')
            seed_everything(EXP_NUM)
            break


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


def train_once(model, optimizer, criterion, train_loader):
    model.train()

    total_loss = 0
    N = 0

    for source, dialect, source_lens, target in train_loader:
        source, dialect, source_lens, target = \
            source.to(DEVICE), dialect.to(DEVICE), source_lens.to(DEVICE), target.to(DEVICE)

        target_in = target[:, :-1]
        source_mask, target_mask, source_padding_mask, target_padding_mask = \
            create_mask(source, target_in)

        # forward
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

        # gradient update
        optimizer.zero_grad()

        target_out = target[:, 1:]
        loss = criterion(
            logits.reshape((-1, logits.shape[-1])),
            target_out.reshape(-1)
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        N += source.shape[0]

    return total_loss / N


def train(conf, model, optimizer, scheduler, criterion, ipa_vocab, dialect_vocab):
    # load datasets
    train_dataset = Dataset(
        get_dataset_path('train'),
        ipa_vocab,
        dialect_vocab
    )
    dev_dataset = Dataset(
        get_dataset_path('dev'),
        ipa_vocab,
        dialect_vocab
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=conf.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=0
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=conf.batch_size,
        shuffle=False,
        collate_fn=dev_dataset.collate_fn,
        num_workers=0
    )

    best_dev_loss = 1e10
    best_edit_distance = 1e10

    for epoch in range(conf.epochs):
        t = time.time()

        train_loss = train_once(model, optimizer, criterion, train_loader)
        dev_loss, edit_distance, dev_accuracy, _ = evaluate(model, criterion, dev_loader, ipa_vocab)

        print(f'< epoch {epoch} >  (elapsed: {time.time() - t:.2f}s)')
        print(f'  * [train]  loss: {train_loss:.6f}')
        dev_result_line = f'  * [ dev ]  loss: {dev_loss:.6f}'
        if edit_distance is not None:
            dev_result_line += f'  ||  ED: {edit_distance}  ||  accuracy: {dev_accuracy}'
        print(dev_result_line)
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            save_model(model, optimizer, conf, ipa_vocab, dialect_vocab, epoch, get_checkpoint_path('loss'))
        if edit_distance < best_edit_distance:
            best_edit_distance = edit_distance
            save_model(model, optimizer, conf, ipa_vocab, dialect_vocab, epoch, get_checkpoint_path('token_edit_distance'))

        print()
        scheduler.step()


def evaluate(model, criterion, loader, ipa_vocab):
    model.eval()

    total_loss = 0
    edit_distance = 0
    n_correct, n_total = 0, 0

    ret_predictions = []
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

            edit_distance += get_edit_distance(gold_char_list, pred_char_list)
            if ''.join(gold_char_list) == ''.join(pred_char_list):
                n_correct += 1
            n_total += 1

    accuracy = n_correct / n_total
    total_loss /= n_total
    edit_distance /= n_total

    return total_loss, edit_distance, accuracy, ret_predictions


def test(criterion, conf):
    # test on best checkpoints
    for metric in conf.checkpoint_metrics:
        checkpoint_path = get_checkpoint_path(metric)
        saved_info = load_model(checkpoint_path)

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

        dev_dataset = Dataset(get_dataset_path('dev'), ipa_vocab, dialect_vocab)
        dev_loader = DataLoader(dev_dataset, batch_size=conf.batch_size, shuffle=False, collate_fn=dev_dataset.collate_fn)

        test_dataset = Dataset(get_dataset_path('test'), ipa_vocab, dialect_vocab)
        test_loader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)

        dev_loss, dev_ed, dev_acc, dev_preds = evaluate(model, criterion, dev_loader, ipa_vocab)
        test_loss, test_ed, test_acc, test_preds = evaluate(model, criterion, test_loader, ipa_vocab)

        print(f'===== <FINAL - best {metric}>  (epoch: {saved_info["epoch"]}) ======')

        print(f'[dev]')
        print(f'  * loss: {dev_loss}')
        print(f'  * edit distance: {dev_ed}')
        print(f'  * accuracy: {dev_acc}')
        print()

        print(f'[test]')
        print(f'  * loss: {test_loss}')
        print(f'  * edit distance: {test_ed}')
        print(f'  * accuracy: {test_acc}')
        print()

        dump_predictions(test_preds, metric, 'test_predictions.tsv')


def dump_predictions(predictions, metric, preds_fname):
    dump_path = os.path.join(WORK_DIR, 'checkpoints', DATASET, f'exp_{EXP_NUM}', f'{metric}_{preds_fname}')
    with open(dump_path, 'w') as fout:
        fout.write('GOLD_SEQUENCE\tPREDICTION\n')
        for gs, pred in predictions:
            fout.write(f'{gs}\t{pred}\n')


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


def save_model(model, optimizer, conf, ipa_vocab, dialect_vocab, epoch, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'conf': conf._asdict(),
        'epoch': epoch,
        'ipa_vocab': ipa_vocab,
        'dialect_vocab': dialect_vocab
    }
    torch.save(save_info, filepath)
    print(f'\t>> saved model to {filepath}')


def load_model(filepath):
    saved_info = torch.load(filepath)
    return saved_info


def main():
    args = parse_args()
    load_environment_variables()
    conf = load_config(os.path.join(CONF_DIR, f'{args.conf}.json'))
    declare_globals(conf)
    setup_experiment()

    train_fpath = get_dataset_path('train')
    ipa_vocab, dialect_vocab = build_vocab(train_fpath)

    model = initialize_model(conf, ipa_vocab, dialect_vocab)

    # prepare for training
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=conf.learning_rate,
        betas=(conf.beta1, conf.beta2),
        eps=1e-9,
        weight_decay=conf.weight_decay
    )
    scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
        optimizer,
        conf.warmup_epochs,
        conf.epochs,
        lr_end=0.000001
    )

    train(conf, model, optimizer, scheduler, criterion, ipa_vocab, dialect_vocab)

    test(criterion, conf)


if __name__ == '__main__':
    main()
