import os
import math
import json
import argparse

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import set_seed, get_device, validate_against_schema, get_iterator, verbose_print, Logger
from data import TextEncoder, get_dataloaders
from models.transformer_models import SingleHeadModel, load_openai_pretrained_model
from evaluate import Evaluator


def score(dataloader, model, verbose, evaluator):
    losses = []
    with torch.no_grad():
        model.eval()
        for x, m in get_iterator(dataloader, verbose):
            lm_logits = model(x)
            loss = evaluator.compute_loss(x, m, lm_logits)
            losses.extend([loss.cpu().item()] * x.shape[0])
    return np.mean(losses)

def run_epoch(train_dataloader, validation_dataloader, model, optimizer, lr_scheduler, scores_per_epoch, verbose, evaluator):
    train_losses = []
    validation_losses = []

    model.train()
    n_updates = 0
    for x, m in get_iterator(train_dataloader, verbose):
        lm_logits = model(x)
        loss = evaluator.compute_loss(x, m, lm_logits)
        loss.backward()
        lr_scheduler.step()
        optimizer.step()
        optimizer.zero_grad()

        n_updates += 1
        if n_updates % math.ceil(float(len(train_dataloader)) / float(scores_per_epoch)) == 0 or n_updates == len(train_dataloader):
            train_loss = score(train_dataloader, model, verbose=verbose, evaluator=evaluator)
            validation_loss = score(validation_dataloader, model, verbose=verbose, evaluator=evaluator)
            train_losses.append(train_loss)
            validation_losses.append(validation_loss)

    return train_losses, validation_losses

def train(train_dataloader, validation_dataloader, model, model_opt, lr_scheduler, hyperparams, evaluator, scores_per_epoch, logger):

    min_loss = float('inf')
    if logger is not None:
        params_directory = os.path.join('params', logger.task_name)
        if not os.path.exists(params_directory):
            os.makedirs(params_directory)
        transformer_path = os.path.join(params_directory, 'transformer.pth')

    for epoch in range(hyperparams["n_iter"]):

        verbose_print(verbose, 'Running epoch {}'.format(epoch))

        train_losses, validation_losses = run_epoch(train_dataloader, validation_dataloader, model, model_opt, lr_scheduler, scores_per_epoch, verbose, evaluator)

        if logger is not None:
            logger.results['train_losses'].extend(train_losses)
            logger.results['validation_losses'].extend(validation_losses)
            logger.log()
            logger.plot()

        verbose_print(verbose, 'Train Loss: {}'.format(train_losses))
        verbose_print(verbose, 'Validation Loss: {}'.format(validation_losses))

        new_loss = np.mean(validation_losses)
        if new_loss < min_loss:
            min_loss = np.mean(validation_losses)
            if logger is not None:
                torch.save(model.transformer.state_dict(), transformer_path)

    if logger is not None and min_loss != new_loss:
        model.transformer.load_state_dict(torch.load(transformer_path))

def test(test_dataloader, model, evaluator, logger):
    verbose_print(verbose, 'Testing')

    test_loss = score(test_dataloader, model, verbose, evaluator)
    verbose_print(verbose, 'Test Loss: {}'.format(test_loss))

    if logger is not None:
        logger.results['test_loss'] = test_loss
        logger.log()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--hyperparams', type=str, default='hyperparams/pretrain.json')
    parser.add_argument('--data_file', type=str, default='/users/data/toronto_book_corpus/abridged_6_to_11_len_books_in_sentences.txt')

    args = parser.parse_args()

    verbose = args.verbose
    if verbose:
        verbose_print(verbose, vars(args))

    hyperparams_path = args.hyperparams
    with open(hyperparams_path, 'r') as hyperparams_file:
        hyperparams = json.load(hyperparams_file)
    validate_against_schema(hyperparams, schema_path='schema/pretrain_schema.json')

    data_file_path = args.data_file

    set_seed(hyperparams['seed'])
    device = get_device(verbose)

    text_encoder = TextEncoder(hyperparams['encoder_path'], hyperparams['bpe_path'])
    train_dataloader, validation_dataloader, test_dataloader = get_dataloaders(data_file_path, text_encoder, hyperparams['test_split'], hyperparams['validation_split'], hyperparams['batch_size'], device, verbose, sequence_dim=hyperparams['max_sequence_dim'])

    max_position_encoding = train_dataloader.dataset.max_position_encoding
    sequence_dim = train_dataloader.dataset.sequence_dim
    vocab_size = len(text_encoder.encoder) + max_position_encoding
    model = SingleHeadModel(hyperparams, vocab_size, sequence_dim)

    # load_openai_pretrained_model(model.transformer, n_ctx=sequence_dim, n_special=2)

    lm_criterion = nn.CrossEntropyLoss(reduction='none')
    evaluator = Evaluator(lm_criterion)

    model_opt = Adam(model.parameters(),
                     lr=hyperparams['lr'],
                     betas=(hyperparams['b1'], hyperparams['b2']),
                     eps=hyperparams['eps'])
    lr_scheduler = CosineAnnealingLR(model_opt, hyperparams["n_iter"] * len(train_dataloader))

    model.to(device)

    scores_per_epoch = hyperparams['scores_per_epoch']

    if args.save:
        data_file_name = os.path.splitext(os.path.split(data_file_path)[1])[0]
        logger = Logger(hyperparams, 'language_modeling__{}'.format(data_file_name), data_file_path, sequence_dim, scores_per_epoch)
    else:
        logger = None

    train(train_dataloader, validation_dataloader, model, model_opt, lr_scheduler, hyperparams, evaluator, scores_per_epoch, logger)
    test(test_dataloader, model, evaluator, logger)
