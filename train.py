import os
import math
import json
import argparse

import numpy as np
import torch
from torch import nn

from utils import set_seed, get_device, validate_against_schema, get_iterator, verbose_print, Logger
from data import TextEncoder, get_dataloaders
from models import FLaME
from evaluate import Evaluator
from opt import OpenAIAdam


def run_epoch(train_val_dataloaders, model, optimizer, evaluator, verbose):
    epoch_size = hyperparams['epoch_size']
    validation_frequency = hyperparams['validation_frequency']
    train_dataloader, validate_training_dataloader, validate_validation_dataloader = train_val_dataloaders

    train_losses = {'total': [], 'flow_negative_log_likelihood': [], 'language_modeling': [], 'distance': []}
    validation_losses = {'total': [], 'flow_negative_log_likelihood': [], 'language_modeling': [], 'distance': []}

    model.train()
    n_updates = 0
    for x, m in get_iterator(train_dataloader, epoch_size, verbose):
        z, logdet, lm_logits = model(x)
        loss, _, _, _ = evaluator.compute_flame_loss(model, x, m, z, logdet, lm_logits)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        n_updates += 1
        if n_updates % validation_frequency == 0:
            train_loss, validation_loss = validate(validate_training_dataloader, validate_validation_dataloader, model)
            train_losses['total'].append(train_loss[0])
            train_losses['flow_negative_log_likelihood'].append(train_loss[1])
            train_losses['language_modeling'].append(train_loss[2])
            train_losses['distance'].append(train_loss[3])
            validation_losses['total'].append(validation_loss[0])
            validation_losses['flow_negative_log_likelihood'].append(validation_loss[1])
            validation_losses['language_modeling'].append(validation_loss[2])
            validation_losses['distance'].append(validation_loss[3])
        if n_updates == epoch_size:
            break

    return train_losses, validation_losses

def validate(validate_training_dataloader, validate_validation_dataloader, model):
    with torch.no_grad():
        model.eval()
        x, m = next(iter(validate_training_dataloader))
        z, logdet, lm_logits = model(x)
        train_losses = evaluator.compute_flame_loss(model, x, m, z, logdet, lm_logits)
        x, m = next(iter(validate_validation_dataloader))
        z, logdet, lm_logits = model(x)
        validation_losses = evaluator.compute_flame_loss(model, x, m, z, logdet, lm_logits)
    model.train()
    return [loss.cpu().item() for loss in train_losses], [loss.cpu().item() for loss in validation_losses]

def train(train_val_dataloaders, model, model_opt, hyperparams, evaluator, logger):

    min_loss = float('inf')

    for epoch in range(hyperparams["n_iter"]):

        verbose_print(verbose, 'Running epoch {}'.format(epoch))

        train_losses, validation_losses = run_epoch(train_val_dataloaders, model, model_opt, evaluator, verbose)

        if logger is not None:
            logger.add_train_val_losses(train_losses, validation_losses)
            logger.log_results()
            logger.plot()
            new_loss = np.mean(validation_losses['total'])
            if new_loss < min_loss:
                min_loss = new_loss
                logger.log_weights(model.state_dict(), 'FLaME.pth')

        verbose_print(verbose, '\nTrain Loss: {}'.format(np.mean(train_losses['total'])))
        verbose_print(verbose, 'Validation Loss: {}\n'.format(np.mean(validation_losses['total'])))

    if logger is not None and min_loss != new_loss:
        model_path = os.path.join(logger.params_directory, 'FLaME.pth')
        model.load_state_dict(torch.load(model_path))

def test(test_dataloader, model, evaluator, logger):
    verbose_print(verbose, 'Testing')

    test_losses = {'total': [], 'flow_negative_log_likelihood': [], 'language_modeling': [], 'distance': []}
    with torch.no_grad():
        model.eval()
        for x, m in get_iterator(test_dataloader, verbose):
            z, logdet, lm_logits = model(x)
            losses = evaluator.compute_flame_loss(model, x, m, z, logdet, lm_logits)
            losses = [[loss.cpu().item()] * x.shape[0] for loss in losses]
            test_losses['total'].extend(losses[0])
            test_losses['flow_negative_log_likelihood'].extend(losses[1])
            test_losses['language_modeling'].extend(losses[2])
            test_losses['distance'].extend(losses[3])
    test_loss = {label: np.mean(test_losses[label]) for label in test_losses}

    verbose_print(verbose, 'Test Loss: {}'.format(test_loss['total']))

    if logger is not None:
        logger.set_test_losses(test_loss)
        logger.log_results()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--hyperparams', type=str, default='hyperparams/train.json')
    parser.add_argument('--data_file', type=str, default='/users/data/toronto_book_corpus/6_to_11_len_books_in_sentences.txt')
    # parser.add_argument('--data_file', type=str, default='/users/data/toronto_book_corpus/abridged_6_to_11_len_books_in_sentences.txt')
    # parser.add_argument('--data_file', type=str, default='test.txt')

    args = parser.parse_args()

    verbose = args.verbose
    if verbose:
        verbose_print(verbose, vars(args))

    hyperparams_path = args.hyperparams
    with open(hyperparams_path, 'r') as hyperparams_file:
        hyperparams = json.load(hyperparams_file)
    validate_against_schema(hyperparams, schema_path='schema/train_schema.json')

    data_file_path = args.data_file

    set_seed(hyperparams['seed'])
    device = get_device(verbose)

    text_encoder = TextEncoder(hyperparams['encoder_path'], hyperparams['bpe_path'])
    dataloaders = get_dataloaders(data_file_path, text_encoder, hyperparams['test_split'],
            hyperparams['validation_split'], hyperparams['batch_size'], device,
            verbose, sequence_dim=hyperparams['max_sequence_dim'])
    train_val_dataloaders = dataloaders[:-1]
    test_dataloader = dataloaders[-1]

    max_position_encoding = test_dataloader.dataset.max_position_encoding
    sequence_dim = test_dataloader.dataset.sequence_dim
    vocab_size = len(text_encoder.encoder) + max_position_encoding
    model = FLaME(hyperparams, vocab_size, sequence_dim)

    if 'pretrained_lm_path' in hyperparams:
        pretrained_lm_path = hyperparams['pretrained_lm_path']
        model.language_model.load_state_dict(torch.load(pretrained_lm_path))

    lm_criterion = nn.CrossEntropyLoss(reduction='none')
    evaluator = Evaluator(lm_criterion, hyperparams['lm_coefficient'],
            hyperparams['distance_coefficient'], hyperparams['distance_metric'])

    model_opt = OpenAIAdam(model.parameters(), lr=hyperparams['lr'], schedule=hyperparams['lr_schedule'],
            warmup=hyperparams['lr_warmup'], t_total=hyperparams['n_iter'] * hyperparams['epoch_size'],
            b1=hyperparams['b1'], b2=hyperparams['b2'], eps=hyperparams['eps'],
            l2=hyperparams['l2'], vector_l2=hyperparams['vector_l2'], max_grad_norm=hyperparams['max_grad_norm'])

    model.to(device)

    if args.save:
        data_file_name = os.path.splitext(os.path.split(data_file_path)[1])[0]
        logger = Logger(hyperparams, 'flow_lm__{}'.format(data_file_name), data_file_path, sequence_dim)
    else:
        logger = None

    train(train_val_dataloaders, model, model_opt, hyperparams, evaluator, logger)
    test(test_dataloader, model, evaluator, logger)
