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
    parser.add_argument('--params', type=str, default='params/language_modeling__6_to_11_len_books_in_sentences_2019-03-08 18:45:14')
    parser.add_argument('--prime', type=str, default='')
    parser.add_argument('--n_iter', type=int, default=10)

    args = parser.parse_args()

    prime = args.prime
    n_iter = args.n_iter
    params_path = args.params
    hyperparams_path = args.hyperparams

    with open(hyperparams_path, 'r') as hyperparams_file:
        hyperparams = json.load(hyperparams_file)
    validate_against_schema(hyperparams, schema_path='schema/pretrain_schema.json')

    # set_seed(hyperparams['seed'])
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
    model.transformer.load_state_dict(torch.load(os.path.join(params_path, 'transformer.pth')))
    model.to(device)
    softmax = Softmax(-1)

    for i in range(n_iter):
        logits = model(X)
        probabilities = softmax(logits[0, encoded_prime_length + i - 1,:len(text_encoder.encoder)])
        indices = torch.multinomial(probabilities, 1).reshape(X.shape[0], 1, 1)
        X[:, encoded_prime_length + i, 0] = indices

    encoded_sequences = X.cpu().numpy()[:, :encoded_prime_length + n_iter, 0]

    for encoded_sequence in encoded_sequences:
        decoded_sequence = [text_encoder.decoder[token] for token in encoded_sequence]
        start = decoded_sequence.index('_start_')
        try:
            end = decoded_sequence.index('_end_')
        except ValueError as e:
            end = None
        print(decoded_sequence[start + 1: end])
