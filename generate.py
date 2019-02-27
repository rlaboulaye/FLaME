import argparse
import json
import os

import numpy as np
import torch
from torch.nn import Softmax

from utils import set_seed, get_device, validate_against_schema
from data import TextEncoder
from models.transformer_models import SingleHeadModel, load_openai_pretrained_model


def encode_prime(text_encoder, prime, sequence_dim):
    encoded_sequence = [text_encoder.start_token] + text_encoder.encode(prime)
    return np.array(encoded_sequence + [0 for i in range(sequence_dim - len(encoded_sequence))]), len(encoded_sequence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperparams', type=str, default='hyperparams/pretrain.json')
    parser.add_argument('--params', type=str, default='params/language_modeling_2019-02-26 18:17:56')
    parser.add_argument('--prime', type=str, default='The cat is')
    parser.add_argument('--n_iter', type=int, default=5)

    args = parser.parse_args()

    prime = args.prime
    n_iter = args.n_iter
    params_path = args.params
    hyperparams_path = args.hyperparams

    with open(hyperparams_path, 'r') as hyperparams_file:
        hyperparams = json.load(hyperparams_file)
    validate_against_schema(hyperparams, schema_path='schema/pretrain_schema.json')

    set_seed(hyperparams['seed'])
    device = get_device()

    log_path = os.path.join('results', os.path.split(params_path)[-1], 'results.json')
    with open(log_path, 'r') as log_file:
        sequence_dim = json.load(log_file)['sequence_dim']

    text_encoder = TextEncoder(hyperparams['encoder_path'], hyperparams['bpe_path'])
    X_value, encoded_prime_length = encode_prime(text_encoder, prime, sequence_dim)

    max_position_encoding = X_value.shape[-1]
    vocab_size = len(text_encoder.encoder) + max_position_encoding

    X_positions = np.arange(len(text_encoder.encoder), len(text_encoder.encoder) + max_position_encoding)
    X = np.zeros(X_value.shape + (2,))
    X[:, 0] = X_value
    X[:, 1] = X_positions
    X = torch.tensor(X, dtype=torch.int64, device=device)
    X = X.unsqueeze(0)

    model = SingleHeadModel(hyperparams, vocab_size, sequence_dim)
    # model.transformer.load_state_dict(torch.load(os.path.join(params_path, 'transformer.pth')))
    # model.lm_head.load_state_dict(torch.load(os.path.join(params_path, 'lm_head.pth')))
    load_openai_pretrained_model(model.transformer, n_ctx=sequence_dim, n_special=2)
    model.to(device)
    softmax = Softmax(-1)

    for i in range(n_iter):
        logits = model(X)
        probabilities = softmax(logits[0, X.shape[1] - n_iter + i - 1])
        indices = torch.multinomial(probabilities, 1).reshape(X.shape[0], 1, 1)
        X[:, X.shape[1] - n_iter + i, 0] = indices

    encoded_sequences = X.cpu().numpy()[:, :encoded_prime_length + n_iter, 0]

    for encoded_sequence in encoded_sequences:
        print(encoded_sequence)
        decoded_sequence = [text_encoder.decoder[token] for token in encoded_sequence]
        print(decoded_sequence)
