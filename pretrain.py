import os
import math
import json
import argparse

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from utils import set_seed, get_device, validate_against_schema, get_iterator, verbose_print, Logger
from data import TextEncoder, get_dataloaders
from models.transformer_models import SingleHeadModel
from evaluate import Evaluator


def score(dataloader, model, verbose, evaluator):
    losses = []
    accuracies = []
    with torch.no_grad():
        model.eval()
        for x, m, y in get_iterator(dataloader, verbose):
            lm_logits, task_logits = model(x)
            double_head_loss, task_loss, lm_loss = evaluator.compute_double_head_loss(x, y, m, lm_logits, task_logits)
            accuracy = evaluator.compute_score(y, task_logits)
            losses.extend([double_head_loss.cpu().item()] * x.shape[0])
            accuracies.extend([accuracy.cpu().item()] * x.shape[0])
    return np.mean(losses), np.mean(accuracies)

def run_epoch(train_dataloader, validation_dataloader, model, optimizer, scores_per_epoch, verbose, evaluator):
    train_losses = []
    train_accuracies = []
    validation_losses = []
    validation_accuracies = []

    model.train()
    n_updates = 0
    for x, m, y in get_iterator(train_dataloader, verbose):
        lm_logits, task_logits = model(x)
        double_head_loss, task_loss, lm_loss = evaluator.compute_double_head_loss(x, y, m, lm_logits, task_logits)
        double_head_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        n_updates += 1
        if n_updates % math.ceil(float(len(train_dataloader)) / float(scores_per_epoch)) == 0 or n_updates == len(train_dataloader):
            train_loss, train_accuracy = score(train_dataloader, model, verbose=verbose, evaluator=evaluator)
            validation_loss, validation_accuracy = score(validation_dataloader, model, verbose=verbose, evaluator=evaluator)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            validation_losses.append(validation_loss)
            validation_accuracies.append(validation_accuracy)

    return train_losses, train_accuracies, validation_losses, validation_accuracies

def train(train_dataloader, validation_dataloader, model, model_opt, logger, hyperparams, evaluator):

    min_loss = float('inf')
    weight_directory = os.path.join('weights', logger.task_name)
    if not os.path.exists(weight_directory):
        os.makedirs(weight_directory)
    transformer_path = os.path.join(weight_directory, 'transformer.pth')
    lm_head_path = os.path.join(weight_directory, 'lm_head.pth')
    task_head_path = os.path.join(weight_directory, 'task_head.pth')

    for epoch in range(hyperparams["n_iter"]):

        verbose_print(verbose, 'Running epoch {}'.format(epoch))

        train_losses, train_accuracies, validation_losses, validation_accuracies = run_epoch(train_dataloader, validation_dataloader, model, model_opt, logger.results['scores_per_epoch'], verbose, evaluator)
        logger.results['train_losses'].extend(train_losses)
        logger.results['train_accuracies'].extend(train_accuracies)
        logger.results['validation_losses'].extend(validation_losses)
        logger.results['validation_accuracies'].extend(validation_accuracies)

        logger.log()
        logger.plot()

        verbose_print(verbose, 'Train Loss: {}'.format(train_losses))
        verbose_print(verbose, 'Train Accuracy: {}'.format(train_accuracies))
        verbose_print(verbose, 'Validation Loss: {}'.format(validation_losses))
        verbose_print(verbose, 'Validation Accuracy: {}'.format(validation_accuracies))

        new_loss = np.mean(validation_losses)
        if new_loss < min_loss:
            min_loss = np.mean(validation_losses)
            torch.save(model.transformer.state_dict(), transformer_path)
            torch.save(model.lm_head.state_dict(), lm_head_path)
            torch.save(model.task_head.state_dict(), task_head_path)

    if min_loss != new_loss:
        model.transformer.load_state_dict(torch.load(transformer_path))
        model.lm_head.load_state_dict(torch.load(lm_head_path))
        model.task_head.load_state_dict(torch.load(task_head_path))

def test(test_dataloader, model, logger, evaluator):
    verbose_print(verbose, 'Testing')

    test_loss, test_accuracy = score(test_dataloader, model, verbose, evaluator)
    logger.results['test_loss'] = test_loss
    logger.results['test_accuracy'] = test_accuracy
    logger.log()

    verbose_print(verbose, 'Test Loss: {}'.format(test_loss))
    verbose_print(verbose, 'Test Accuracy: {}'.format(test_accuracy))

def load_openai_pretrained_model(model, n_ctx=-1, n_special=-1, n_transfer=12,
        n_embd=768, path='./params/', verbose=True):
    import re
    # Load weights from TF model
    verbose_print(verbose, "Loading weights...")
    names = json.load(open(path + 'parameters_names.json'))
    shapes = json.load(open(path + 'params_shapes.json'))
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    init_params = [np.load(path + 'params_{}.npy'.format(n)) for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
    if n_ctx > 0:
        init_params[0] = init_params[0][:n_ctx]
    if n_special > 0:
        init_params[0] = np.concatenate(
            [init_params[1],
            (np.random.randn(n_special, n_embd) * 0.02).astype(np.float32),
            init_params[0]
            ], 0)
    else:
        init_params[0] = np.concatenate(
            [init_params[1],
            init_params[0]
            ], 0)
    del init_params[1]
    if n_transfer == -1:
        n_transfer = 0
    else:
        n_transfer = 1 + n_transfer * 12
    init_params = [arr.squeeze() for arr in init_params]

    try:
        assert model.embed.weight.shape == init_params[0].shape
    except AssertionError as e:
        e.args += (model.embed.weight.shape, init_params[0].shape)
        raise

    model.embed.weight.data = torch.from_numpy(init_params[0])

    for name, ip in zip(names[1:n_transfer], init_params[1:n_transfer]):
        name = name[6:]  # skip "model/"
        assert name[-2:] == ":0"
        name = name[:-2]
        name = name.split('/')
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+\d+', m_name):
                l = re.split(r'(\d+)', m_name)
            else:
                l = [m_name]
            pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        try:
            if name[-1] == 'w':
                ip = ip.T
            assert pointer.shape == ip.shape
        except AssertionError as e:
            e.args += (pointer.shape, ip.shape)
            raise
        pointer.data = torch.from_numpy(ip)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--hyperparams', type=str, default='hyperparams/pretrain.json')
    # parser.add_argument('--data_file', type=str, default='/users/data/toronto_book_corpus/abridged_books_in_sentences.txt')
    parser.add_argument('--data_file', type=str, default='test.txt')

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
    train_dataloader, validation_dataloader, test_dataloader = get_dataloaders(data_file_path, text_encoder, hyperparams['test_split'], hyperparams['validation_split'], hyperparams['batch_size'], device, verbose, sequence_dim=hyperparams['sequence_dim'])

    max_position_encoding = train_dataloader.dataset.max_position_encoding
    sequence_dim = train_dataloader.dataset.sequence_dim
    vocab_size = len(text_encoder.encoder) + max_position_encoding
    model = SingleHeadModel(hyperparams, vocab_size, sequence_dim)

    load_openai_pretrained_model(model.transformer, n_ctx=sequence_dim, n_special=2, verbose=verbose)

    lm_criterion = nn.CrossEntropyLoss(reduction='none')

    model_opt = Adam(model.parameters(),
                     lr=hyperparams['lr'],
                     betas=(hyperparams['b1'], hyperparams['b2']),
                     eps=hyperparams['eps'])

    model.to(device)

    scores_per_epoch = hyperparams['scores_per_epoch']
    logger = Logger(hyperparams, 'language_modeling', scores_per_epoch)

    # train(train_dataloader, validation_dataloader, dh_model, model_opt, logger, hyperparams, evaluator)
    # test(test_dataloader, dh_model, logger, evaluator)
