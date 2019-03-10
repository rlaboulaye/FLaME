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
from opt import OpenAIAdam


def run_epoch(train_val_dataloaders, model, optimizer, lr_scheduler, evaluator, verbose):
    epoch_size = hyperparams['epoch_size']
    validation_frequency = hyperparams['validation_frequency']
    train_losses = []
    validation_losses = []
    train_dataloader, validate_training_dataloader, validate_validation_dataloader = train_val_dataloaders

    model.train()
    n_updates = 0
    for x, m in get_iterator(train_dataloader, epoch_size, verbose):
        lm_logits = model(x)
        loss = evaluator.compute_lm_loss(x, m, lm_logits)
        loss.backward()
        # lr_scheduler.step()
        optimizer.step()
        optimizer.zero_grad()

        n_updates += 1
        if n_updates % validation_frequency == 0:
            train_loss, validation_loss = validate(validate_training_dataloader, validate_validation_dataloader, model)
            train_losses.append(train_loss)
            validation_losses.append(validation_loss)
        if n_updates == epoch_size:
            break

    return train_losses, validation_losses

def validate(validate_training_dataloader, validate_validation_dataloader, model):
    with torch.no_grad():
        model.eval()
        x, m = next(iter(validate_training_dataloader))
        train_loss = evaluator.compute_lm_loss(x, m, model(x)).cpu().item()
        x, m = next(iter(validate_validation_dataloader))
        validation_loss = evaluator.compute_lm_loss(x, m, model(x)).cpu().item()
    model.train()
    return train_loss, validation_loss

def train(train_val_dataloaders, model, model_opt, lr_scheduler, hyperparams, evaluator, logger):

    min_loss = float('inf')

    for epoch in range(hyperparams["n_iter"]):

        verbose_print(verbose, 'Running epoch {}'.format(epoch))

        train_losses, validation_losses = run_epoch(train_val_dataloaders, model, model_opt, lr_scheduler, evaluator, verbose)

        if logger is not None:
            logger.results['train_losses'].extend(train_losses)
            logger.results['validation_losses'].extend(validation_losses)
            logger.log_results()
            logger.plot()
            new_loss = np.mean(validation_losses)
            if new_loss < min_loss:
                min_loss = np.mean(validation_losses)
                logger.log_weights(model.transformer.state_dict(), 'transformer.pth')

        verbose_print(verbose, '\nTrain Loss: {}'.format(np.mean(train_losses)))
        verbose_print(verbose, 'Validation Loss: {}\n'.format(np.mean(validation_losses)))

    if logger is not None and min_loss != new_loss:
        transformer_path = os.path.join(logger.params_directory, 'transformer.pth')
        model.transformer.load_state_dict(torch.load(transformer_path))

def test(test_dataloader, model, evaluator, logger):
    verbose_print(verbose, 'Testing')

    losses = []
    with torch.no_grad():
        model.eval()
        for x, m in get_iterator(test_dataloader, verbose):
            lm_logits = model(x)
            loss = evaluator.compute_lm_loss(x, m, lm_logits)
            losses.extend([loss.cpu().item()] * x.shape[0])
    test_loss = np.mean(losses)

    verbose_print(verbose, 'Test Loss: {}'.format(test_loss))

    if logger is not None:
        logger.results['test_loss'] = test_loss
        logger.log_results()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--hyperparams', type=str, default='hyperparams/pretrain.json')
    parser.add_argument('--data_file', type=str, default='/users/data/toronto_book_corpus/6_to_11_len_books_in_sentences.txt')

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
    dataloaders = get_dataloaders(data_file_path, text_encoder, hyperparams['test_split'],
            hyperparams['validation_split'], hyperparams['batch_size'], device,
            verbose, sequence_dim=hyperparams['max_sequence_dim'])
    train_val_dataloaders = dataloaders[:-1]
    test_dataloader = dataloaders[-1]

    max_position_encoding = test_dataloader.dataset.max_position_encoding
    sequence_dim = test_dataloader.dataset.sequence_dim
    vocab_size = len(text_encoder.encoder) + max_position_encoding
    model = SingleHeadModel(hyperparams, vocab_size, sequence_dim)

    # load_openai_pretrained_model(model.transformer, n_ctx=sequence_dim, n_special=2)

    lm_criterion = nn.CrossEntropyLoss(reduction='none')
    evaluator = Evaluator(lm_criterion)

    # model_opt = Adam(model.parameters(),
    #                  lr=hyperparams['lr'],
    #                  betas=(hyperparams['b1'], hyperparams['b2']),
    #                  eps=hyperparams['eps'])
    # lr_scheduler = CosineAnnealingLR(model_opt, hyperparams['n_iter'] * hyperparams['epoch_size'])

    #
    model_opt = OpenAIAdam(model.parameters(), lr=hyperparams['lr'], schedule=hyperparams['lr_schedule'],
            warmup=hyperparams['lr_warmup'], t_total=hyperparams['n_iter'] * hyperparams['epoch_size'],
            b1=hyperparams['b1'], b2=hyperparams['b2'], eps=hyperparams['eps'],
            l2=hyperparams['l2'], vector_l2=hyperparams['vector_l2'], max_grad_norm=hyperparams['max_grad_norm'])
    lr_scheduler = None
    #

    model.to(device)

    if args.save:
        data_file_name = os.path.splitext(os.path.split(data_file_path)[1])[0]
        logger = Logger(hyperparams, 'language_modeling__{}'.format(data_file_name), data_file_path, sequence_dim)
    else:
        logger = None

    train(train_val_dataloaders, model, model_opt, lr_scheduler, hyperparams, evaluator, logger)
    test(test_dataloader, model, evaluator, logger)
