import math
import os

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils import data


class Dataset(data.Dataset):

    def __init__(self, device, vocab_size, instances, masks, max_position_encoding=512):
        self.device = device
        self.instances = instances
        self.masks = masks
        self.sequence_dim = self.instances.shape[-1]
        self.max_position_encoding = min(self.sequence_dim, max_position_encoding)
        self.X_positions = np.arange(vocab_size, vocab_size + self.max_position_encoding)
        while self.X_positions.shape[0] < self.sequence_dim:
            self.X_positions = np.concatenate([self.X_positions,
                np.arange(vocab_size, vocab_size + min(self.sequence_dim - self.X_positions.shape[0], max_position_encoding))], 0)

    def __len__(self):
        return self.instances.shape[0]

    def __getitem__(self, index):
        # M is a mask
        X_tokens = self.instances[index]
        X_value = np.zeros(X_tokens.shape + (2,))
        X_value[:, 0] = X_tokens
        X_value[:, 1] = self.X_positions
        X = torch.tensor(X_value, dtype=torch.int64, device=self.device)
        M = torch.tensor(self.masks[index], dtype=torch.float32, device=self.device)
        return X, M


def load_document_series(path):
    with open(path, 'r') as f:
        document_series = pd.Series(f.read().split('\n'))
    filter = document_series.apply(lambda x: len(x)) > 0
    return document_series[filter]


def split_series(series, split=.2):
    split_s2 = series.sample(frac=split, replace=False)
    split_s1 = series.drop(split_s2.index)
    return split_s1, split_s2


def encode_documents(document_series, text_encoder, verbose, sequence_dim):
    tqdm.pandas(disable=not verbose, ncols=150, desc='Encoding document for each instance')
    encoded_series = document_series.progress_apply(text_encoder.encode)
    tqdm.pandas(disable=not verbose, ncols=150, desc='Appending special tokens')
    num_tokens = 2
    doc_length = sequence_dim - num_tokens if sequence_dim is not None else None
    return encoded_series.progress_apply(lambda x: np.array([text_encoder.start_token] + x[:doc_length] + [text_encoder.end_token]))


def get_document_matrix(documents_series, max_sequence_length):
    document_matrix = np.stack(documents_series.apply(lambda x: np.pad(x, (0, max_sequence_length - len(x)), mode='constant')).values)
    mask_matrix = np.stack(documents_series.apply(lambda x: np.pad(np.ones(len(x)), (0, max_sequence_length - len(x)), mode='constant')).values)
    return document_matrix, mask_matrix


def get_dataloaders(file_path, text_encoder, test_split, validation_split, batch_size, device, verbose, sequence_dim=None):

    document_series = load_document_series(file_path)
    train_val_series, test_series = split_series(document_series, test_split)
    train_series, validation_series = split_series(train_val_series, validation_split)

    train_encoded_series = encode_documents(train_series, text_encoder, verbose, sequence_dim)
    validation_encoded_series = encode_documents(validation_series, text_encoder, verbose, sequence_dim)
    test_encoded_series = encode_documents(test_series, text_encoder, verbose, sequence_dim)

    max_sequence_length = max(
        train_encoded_series.apply(lambda x: len(x)).max(),
        validation_encoded_series.apply(lambda x: len(x)).max(),
        test_encoded_series.apply(lambda x: len(x)).max())
    if sequence_dim is not None:
        max_sequence_length = min(sequence_dim, max_sequence_length)

    train_document_matrix, train_mask_matrix = get_document_matrix(train_encoded_series, max_sequence_length)
    validation_document_matrix, validation_mask_matrix = get_document_matrix(validation_encoded_series, max_sequence_length)
    test_document_matrix, test_mask_matrix = get_document_matrix(test_encoded_series, max_sequence_length)

    vocab_size = len(text_encoder.encoder)

    train_set = Dataset(device, vocab_size, train_document_matrix, train_mask_matrix)
    validation_set = Dataset(device, vocab_size, validation_document_matrix, validation_mask_matrix)
    test_set = Dataset(device, vocab_size, test_document_matrix, test_mask_matrix)
    train_data_params = {
        'batch_size': batch_size,
        'shuffle': True
    }
    test_data_params = {
        'batch_size': 8 * batch_size,
        'shuffle': True
    }
    train_dataloader = data.DataLoader(train_set, **train_data_params)
    validate_training_dataloader = data.DataLoader(train_set, **test_data_params)
    validate_validation_dataloader = data.DataLoader(validation_set, **test_data_params)
    test_dataloader = data.DataLoader(test_set, **test_data_params)
    return train_dataloader, validate_training_dataloader, validate_validation_dataloader, test_dataloader
