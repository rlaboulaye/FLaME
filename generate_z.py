import argparse
import json
import os

import numpy as np
import torch

from utils import set_seed, get_device, validate_against_schema
from data import TextEncoder
from models import FLaME


def encode_prime(text_encoder, prime):
    encoded_sequence = [text_encoder.start_token] + text_encoder.encode(prime)
    return np.array(encoded_sequence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperparams', type=str, default='hyperparams/train.json')
    parser.add_argument('--params', type=str, default='params/flow_lm__abridged_6_to_11_len_books_in_sentences_2019-03-18 14:50:28')
    parser.add_argument('--prime', type=str, default='')
    parser.add_argument('--max_length', type=int, default=16)

    args = parser.parse_args()

    prime = args.prime
    max_length = args.max_length
    params_path = args.params
    hyperparams_path = args.hyperparams

    with open(hyperparams_path, 'r') as hyperparams_file:
        hyperparams = json.load(hyperparams_file)
    validate_against_schema(hyperparams, schema_path='schema/train_schema.json')

    # set_seed(hyperparams['seed'])
    device = get_device()

    text_encoder = TextEncoder(hyperparams['encoder_path'], hyperparams['bpe_path'])
    x_value = encode_prime(text_encoder, prime)

    ###
    # once i retrain lm, use n_ctx from hyperparams instead of max_sequence_dim
    # vocab_size = len(text_encoder.encoder) + hyperparams['n_ctx']
    vocab_size = len(text_encoder.encoder) + hyperparams['max_sequence_dim']
    ###
    position_token = len(text_encoder.encoder)

    x_positions = np.arange(position_token, position_token + x_value.shape[-1])
    x = np.zeros(x_value.shape + (2,))
    x[:, 0] = x_value
    x[:, 1] = x_positions
    x = torch.tensor(x, dtype=torch.int64, device=device)
    x = x.unsqueeze(0)

    model = FLaME(hyperparams, vocab_size)
    model.load_state_dict(torch.load(os.path.join(params_path, 'FLaME.pth')))
    model.to(device)
    model.eval()

    x = model.generate(x, position_token, max_length=max_length)

    encoded_sequences = x.cpu().numpy()[:, :max_length, 0]

    for encoded_sequence in encoded_sequences:
        decoded_sequence = [text_encoder.decoder[token] for token in encoded_sequence]
        start = decoded_sequence.index('_start_')
        try:
            end = decoded_sequence.index('_end_')
        except ValueError as e:
            end = None
        print(decoded_sequence[start + 1: end])
