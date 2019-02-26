import math

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
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
        X_value[:, :, 0] = X_tokens
        X_value[:, :, 1] = self.X_positions
        X = torch.tensor(X_value, dtype=torch.int64, device=self.device)
        M = torch.tensor(self.masks[index], dtype=torch.float32, device=self.device)
        return X, M


def get_dataloaders(text_encoder, test_split, validation_split, batch_size, device, verbose, train_file, test_file=None, sequence_dim=None):

    train_file = task['train_file']
    train_val_dataframe = load_dataframe(train_file['file_path'], train_file['file_type'], train_file['file_header'])
    if 'test_file' in task:
        test_file = task['test_file']
        test_dataframe = load_dataframe(test_file['file_path'], test_file['file_type'], test_file['file_header'])
    else:
        train_val_dataframe, test_dataframe = split_dataframe(train_val_dataframe, test_split)
    train_dataframe, validation_dataframe = split_dataframe(train_val_dataframe, validation_split)

    raw_documents = task["documents"]
    train_documents_dataframe, document_structure = create_documents(train_dataframe, raw_documents, text_encoder, verbose, sequence_dim)
    validation_documents_dataframe, _ = create_documents(validation_dataframe, raw_documents, text_encoder, verbose, sequence_dim)
    test_documents_dataframe, _ = create_documents(test_dataframe, raw_documents, text_encoder, verbose, sequence_dim)

    max_sequence_length = max(
        max([train_documents_dataframe[column].apply(lambda x: len(x)).max() for column in train_documents_dataframe.columns]),
        max([validation_documents_dataframe[column].apply(lambda x: len(x)).max() for column in validation_documents_dataframe.columns]),
        max([test_documents_dataframe[column].apply(lambda x: len(x)).max() for column in test_documents_dataframe.columns]))
    if sequence_dim is not None:
        max_sequence_length = min(sequence_dim, max_sequence_length)

    train_document_matrix, train_mask_matrix = get_document_matrix(train_documents_dataframe, max_sequence_length)
    train_matrices = (train_document_matrix, train_mask_matrix)
    validation_document_matrix, validation_mask_matrix = get_document_matrix(validation_documents_dataframe, max_sequence_length)
    validation_matrices = (validation_document_matrix, validation_mask_matrix)
    test_document_matrix, test_mask_matrix = get_document_matrix(test_documents_dataframe, max_sequence_length)
    test_matrices = (test_document_matrix, test_mask_matrix)

    target_type = task['target']['target_type']
    target_index = task['target']['column_index']

    encoder = LabelEncoder()
    encoder.fit_transform(np.concatenate([train_dataframe[train_dataframe.columns[target_index]].unique(),
        validation_dataframe[validation_dataframe.columns[target_index]].unique(),
        test_dataframe[test_dataframe.columns[target_index]].unique()]))

    train_target_matrix, target_encoders = get_target_matrix(train_dataframe, target_index, target_type, [encoder])
    train_matrices += (train_target_matrix,)

    validation_target_matrix, _ = get_target_matrix(validation_dataframe, target_index, target_type, target_encoders)
    validation_matrices += (validation_target_matrix,)

    test_target_matrix, _ = get_target_matrix(test_dataframe, target_index, target_type, target_encoders)
    test_matrices += (test_target_matrix,)

    vocab_size = len(text_encoder.encoder)

    train_set = Dataset(device, target_type, vocab_size, *train_matrices)
    validation_set = Dataset(device, target_type, vocab_size, *validation_matrices)
    test_set = Dataset(device, target_type, vocab_size, *test_matrices)
    data_params = {
        'batch_size': batch_size,
        'shuffle': True
    }
    return data.DataLoader(train_set, **data_params), data.DataLoader(validation_set, **data_params), data.DataLoader(test_set, **data_params), document_structure


def load_dataframe(path, file_type, has_header):
    if file_type == 'csv':
        separator = ','
    elif file_type == 'tsv':
        separator = '\t'
    else:
        raise NotImplementedError('Cannot load {} file type'.format(file_type))
    if has_header:
        return pd.read_csv(path, sep=separator, header=0)
    else:
        return pd.read_csv(path, sep=separator)


def get_target_matrix(dataframe, target_col_index, target_type, encoders=None):
    if target_type == 'regression':
        return dataframe[dataframe.columns[target_col_index]].values, None

    targets = []
    if encoders is None:
        target_col = dataframe[dataframe.columns[target_col_index]]
        encoder = LabelEncoder()
        targets.append(encoder.fit_transform(target_col).reshape(-1, 1))
        encoders = (encoder,)
    else:
        for encoder, index in zip(encoders, [target_col_index]):
            targets.append(encoder.transform(dataframe[dataframe.columns[index]]).reshape(-1,1))
    target_matrix = np.concatenate(targets, axis=1)
    if target_matrix.shape[1] == 1:
        target_matrix = target_matrix.reshape(-1)
    return target_matrix, encoders


def get_document_matrix(documents_dataframe, max_sequence_length):
    document_matrices = [np.stack(documents_dataframe[column].apply(lambda x: np.pad(x, (0, max_sequence_length - len(x)), mode='constant')).values) for column in documents_dataframe.columns]
    mask_matrices = [np.stack(documents_dataframe[column].apply(lambda x: np.pad(np.ones(len(x)), (0, max_sequence_length - len(x)), mode='constant')).values) for column in documents_dataframe.columns]
    document_matrix = np.concatenate([document_matrix.reshape(-1, 1, max_sequence_length) for document_matrix in document_matrices], axis=1)
    mask_matrix = np.concatenate([mask_matrix.reshape(-1, 1, max_sequence_length) for mask_matrix in mask_matrices], axis=1)
    return document_matrix, mask_matrix


def create_documents(dataframe, documents, text_encoder, verbose, sequence_dim):
    assert len(documents) == 1 or len(documents) ==2
    if len(documents) == 1:
        return create_one_document(dataframe, documents['primary_document'], text_encoder, verbose, sequence_dim)
    else:
        assert len(documents['associated_documents']) > 0
        if len(documents['associated_documents']) == 1:
            return create_one_to_one_document(dataframe, documents["primary_document"], documents["associated_documents"][0], text_encoder, verbose, sequence_dim)
        else:
            return create_one_to_many_document(dataframe, documents["primary_document"], documents["associated_documents"], text_encoder, verbose, sequence_dim)


def create_one_document(dataframe, document, text_encoder, verbose, sequence_dim):
    document_dataframe = encode_documents(dataframe, [document], text_encoder, verbose)
    tqdm.pandas(disable=not verbose, ncols=150, desc='Appending special tokens to {} document(s) for each instance'.format(document_dataframe.shape[1] - 1))

    tqdm.pandas(disable=not verbose, ncols=150, desc='Appending special tokens to 1 document for each instance')
    assert (document_dataframe.shape[1] == 1)
    num_tokens = 2
    doc_length = sequence_dim - num_tokens if sequence_dim is not None else None
    return document_dataframe.progress_apply(
        lambda x: pd.Series([[text_encoder.start_token] + x[0][:doc_length] + [text_encoder.classify_token]]), axis=1), "one"


def create_one_to_one_document(dataframe, doc1, doc2, text_encoder, verbose, sequence_dim):
    documents_dataframe = encode_documents(dataframe, [doc1, doc2], text_encoder, verbose)
    tqdm.pandas(disable=not verbose, ncols=150, desc='Appending special tokens to {} document(s) for each instance'.format(documents_dataframe.shape[1] - 1))

    tqdm.pandas(disable=not verbose, ncols=150, desc='Appending special tokens to 1 document for each instance')
    num_tokens = 3
    max_len = sequence_dim - num_tokens if sequence_dim is not None else None
    doc1_length = math.ceil(max_len / 2) if sequence_dim is not None else None
    doc2_length = math.floor(max_len / 2) if sequence_dim is not None else None
    return documents_dataframe.progress_apply(
        lambda x: [text_encoder.start_token] + x[documents_dataframe.columns[0]][:doc1_length] + [
            text_encoder.delimeter_token] + x[documents_dataframe.columns[1]][:doc2_length] + [
                      text_encoder.classify_token], axis=1).to_frame(), "one_to_one"


def create_one_to_many_document(dataframe, primary_doc, secondary_docs, text_encoder, verbose, sequence_dim):
    documents_dataframe = encode_documents(dataframe, [primary_doc] + secondary_docs, text_encoder, verbose)
    tqdm.pandas(disable=not verbose, ncols=150, desc='Appending special tokens to {} document(s) for each instance'.format(documents_dataframe.shape[1] - 1))

    multiple_choice_documents = []
    common_column_name = documents_dataframe.columns[0]

    if sequence_dim is not None:
        num_tokens = 3
        max_len = sequence_dim - num_tokens
        documents_dataframe['scale'] = pd.Series(
            documents_dataframe.apply(lambda x: max_len / (len(x[0]) + max([len(y) for y in x[1:]])), axis=1),
            index=documents_dataframe.index)
        scale_column_name = documents_dataframe.columns[-1]

        for choice_column_name in documents_dataframe.columns[1:-1]:
            multiple_choice_documents.append(
                documents_dataframe[[scale_column_name, common_column_name, choice_column_name]].progress_apply(
                    lambda x:
                    [text_encoder.start_token] +
                    x[common_column_name][:math.floor(len(x[common_column_name]) * x[scale_column_name])] +
                    [text_encoder.delimeter_token] +
                    x[choice_column_name][:max_len - math.floor(len(x[common_column_name]) * x[scale_column_name])] +
                    [text_encoder.classify_token], axis=1))
    else:
        for choice_column_name in documents_dataframe.columns[1:]:
            multiple_choice_documents.append(
                documents_dataframe[[common_column_name, choice_column_name]].progress_apply(
                    lambda x:
                    [text_encoder.start_token] +
                    x[common_column_name] +
                    [text_encoder.delimeter_token] +
                    x[choice_column_name] +
                    [text_encoder.classify_token], axis=1))

    return pd.concat(multiple_choice_documents, axis=1), "one_to_many"


def encode_documents(dataframe, documents, text_encoder, verbose):
    encoded_documents = []
    for document_index, document in enumerate(documents):
        tqdm.pandas(disable=not verbose, ncols=150,
                    desc='Creating document {} of {} for each instance'.format(document_index + 1, len(documents)))
        document_dataframe = dataframe[dataframe.columns[document['column_indices']]].progress_apply(
            lambda x: ' '.join(x), axis=1)
        tqdm.pandas(disable=not verbose, ncols=150,
                    desc='Encoding document {} of {} for each instance'.format(document_index + 1, len(documents)))
        encoded_documents.append(document_dataframe.progress_apply(text_encoder.encode))
    return pd.concat(encoded_documents, axis=1)

def create_multiple_choice_documents(documents_dataframe, sequence_dim, text_encoder):
    multiple_choice_documents = []
    common_column_name = documents_dataframe.columns[0]

    if sequence_dim is not None:
        num_tokens = 4
        max_len = sequence_dim - num_tokens
        documents_dataframe['scale'] = pd.Series(
            documents_dataframe.apply(lambda x: max_len / (len(x[0]) + max([len(y) for y in x[1:]])), axis=1),
            index=documents_dataframe.index)
        scale_column_name = documents_dataframe.columns[-1]

        for choice_column_name in documents_dataframe.columns[1:-1]:
            multiple_choice_documents.append(
                documents_dataframe[[scale_column_name, common_column_name, choice_column_name]].progress_apply(
                    lambda x:
                    [text_encoder.start_token] +
                    x[common_column_name][:math.floor(len(x[common_column_name]) * x[scale_column_name])] +
                    [text_encoder.delimeter_token] +
                    x[choice_column_name][:max_len - math.floor(len(x[common_column_name]) * x[scale_column_name])] +
                    [text_encoder.classify_token], axis=1))
    else:
        for choice_column_name in documents_dataframe.columns[1:]:
            multiple_choice_documents.append(
                documents_dataframe[[common_column_name, choice_column_name]].progress_apply(
                    lambda x:
                    [text_encoder.start_token] +
                    x[common_column_name] +
                    [text_encoder.delimeter_token] +
                    x[choice_column_name] +
                    [text_encoder.classify_token], axis=1))

    return pd.concat(multiple_choice_documents, axis=1)

def split_dataframe(dataframe, split=.2):
    split_df2 = dataframe.sample(frac=split, replace=False)
    split_df1 = dataframe.drop(split_df2.index)
    return split_df1, split_df2
