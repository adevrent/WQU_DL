import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import json

def load_hyperparams(hyperparams_filepath="hyperparams.json"):
    with open (hyperparams_filepath) as r:
        hyperparams = json.load(r)

    for key, value in hyperparams.items():
        hyperparams[key] = int(value)

    return hyperparams.values()

M, K, H, n, batch_size, num_epochs = load_hyperparams()

def read_csv(filepath, date_col, value_col, start_date, end_date):
    """
    Parameters:
    filepath (str): Path to the CSV file
    date_col (str): Name of the column containing the date
    value_col (str): Name of the column containing the time series values
    """
    df = pd.read_csv(filepath, header=0, index_col=date_col, parse_dates=[date_col])
    df.index = pd.to_datetime(df.index, unit='s')

    df = df.loc[start_date:end_date, :]

    series = df.loc[:, value_col]
    series.fillna(method='ffill', inplace=True)

    return series

def featurize_series_NBEATS(series, H, n):
    """
    Parameters:
    series (pd.Series): Time series to convert to PyTorch tensor
    H (int): Forecast horizon (number of future time steps to predict)
    n (int): How many times the forecast horizon to look back
    """
    series = series.values.astype(np.float32).squeeze()

    # Create the feature tensor X by rolling the series n times
    num_rows = len(series) - H*n - H + 1
    X = np.zeros((num_rows, H*n))
    Y = np.zeros((num_rows, H))
    for i in range(num_rows):
        X[i, :] = series[i : i + H*n]
        Y[i, :] = series[i + H*n : i + H*n + H]

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    return X, Y

def train_val_test_split(X, Y, val_size=0.2, test_size=0.1):
    """
    Parameters:
    X (torch.Tensor): Input feature tensor
    Y (torch.Tensor): Target tensor
    val_size (float): Proportion of the data to use for validation
    """
    num_train = int((1 - (val_size + test_size)) * len(X))
    num_val = int(val_size * len(X))
    X_train, Y_train = X[:num_train], Y[:num_train]
    X_val, Y_val = X[num_train:num_train+num_val], Y[num_train:num_train+num_val]
    X_test, Y_test = X[num_train+num_val:], Y[num_train+num_val:]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def create_data_loaders(X_train, Y_train, X_val, Y_val, X_test, Y_test, batch_size):
    """
    Parameters:
    X_train (torch.Tensor): Input feature tensor for training
    Y_train (torch.Tensor): Target tensor for training
    X_val (torch.Tensor): Input feature tensor for validation
    Y_val (torch.Tensor): Target tensor for validation
    batch_size (int): Number of samples
    """
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset   = TensorDataset(X_val, Y_val)
    test_dataset  = TensorDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def NBEATS_Data_Loader(series, H, n, val_size, test_size, batch_size):
    """
    Parameters:
    series (pd.Series): Univariate time series to convert to PyTorch tensor
    H (int): Forecast horizon (number of future time steps to predict)
    n (int): How many times the forecast horizon to look back
    val_size (float): Proportion of the data to use for validation
    batch_size (int): Number of samples
    """
    X, Y = featurize_series_NBEATS(series, H, n)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = train_val_test_split(X, Y, val_size, test_size)
    train_loader, val_loader, test_loader = create_data_loaders(X_train, Y_train, X_val, Y_val, X_test, Y_test, batch_size)

    return train_loader, val_loader, test_loader