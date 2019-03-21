import argparse
import json
import os

import numpy as np
import torch

from utils import set_seed, get_device, validate_against_schema
from data import TextEncoder
from models import FLaME


def encode_sequences(text_encoder, sequences):
    sequences = sequences.split('_end_')
    if sequences[-1] != '':
        raise ValueError('Sequences must end with end token: _end_')
    encoded_sequences = [[text_encoder.start_token] + text_encoder.encode(sequence) + [text_encoder.end_token] for sequence in sequences[:-1]]
    max_sequence_length = max([len(encoded_sequence) for encoded_sequence in encoded_sequences])
    return np.stack([np.array(encoded_sequence + [0 for i in range(max_sequence_length - len(encoded_sequence))]) for encoded_sequence in encoded_sequences], axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperparams', type=str, default='hyperparams/train.json')
    parser.add_argument('--params', type=str, default='params/flow_lm__6_to_11_len_books_in_sentences_2019-03-21 03:06:17')
    parser.add_argument('--sequences', type=str, default='i fell down._end_burp highway down._end_')

    args = parser.parse_args()

    sequences = args.sequences
    params_path = args.params
    hyperparams_path = args.hyperparams

    with open(hyperparams_path, 'r') as hyperparams_file:
        hyperparams = json.load(hyperparams_file)
    validate_against_schema(hyperparams, schema_path='schema/train_schema.json')

    # set_seed(hyperparams['seed'])
    device = get_device()

    text_encoder = TextEncoder(hyperparams['encoder_path'], hyperparams['bpe_path'])
    x_value = encode_sequences(text_encoder, sequences)
    batch_size = x_value.shape[0]

    ###
    # once i retrain lm, use n_ctx from hyperparams instead of max_sequence_dim
    # vocab_size = len(text_encoder.encoder) + hyperparams['n_ctx']
    vocab_size = len(text_encoder.encoder) + hyperparams['max_sequence_dim']
    ###
    position_token = len(text_encoder.encoder)

    x_positions = np.arange(position_token, position_token + x_value.shape[-1])
    x = np.zeros(x_value.shape + (2,))
    x[:, :, 0] = x_value
    x[:, :, 1] = x_positions
    x = torch.tensor(x, dtype=torch.int64, device=device)

    model = FLaME(hyperparams, vocab_size)
    model.load_state_dict(torch.load(os.path.join(params_path, 'FLaME.pth')))
    model.to(device)
    model.eval()

    with torch.no_grad():
        z, _, _ = model(x)
        z = z.view(batch_size, -1, hyperparams['n_embd'])
        #
        log_probs = model.prior.log_prob(z)
        print(log_probs[:, -3])
        #
    #     x_start = np.zeros((z.shape[0],) + (1, 2))
    #     x_start[:, :, 0] = np.array([text_encoder.start_token] * z.shape[0]).reshape(-1, 1)
    #     x_start[:, :, 1] = len(text_encoder.encoder)
    #     x_start = torch.tensor(x_start, dtype=torch.int64, device=device)
    #     x = model.generate(x_start, position_token, z=z.mean(dim=-2), max_length=16)
    #     encoded_sequences = x.cpu().numpy()[:, :, 0]

    # for encoded_sequence in encoded_sequences:
    #     decoded_sequence = [text_encoder.decoder[token] for token in encoded_sequence]
    #     start = decoded_sequence.index('_start_')
    #     try:
    #         end = decoded_sequence.index('_end_')
    #     except ValueError as e:
    #         end = None
    #     print(decoded_sequence[start + 1: end])
