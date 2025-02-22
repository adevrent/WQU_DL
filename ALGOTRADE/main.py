import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch import nn

import dataloaders
from NBEATS import *

data_filepath = r"C:\Users\Atakan\atakan_python\WQU_DL\ALGOTRADE\static_files\btc_1min.csv"
start_date = "2019-01-01"
end_date = "2021-03-01"

# Read the CSV file
series = dataloaders.read_csv(data_filepath, date_col="Timestamp", value_col="Close", start_date=start_date, end_date=end_date)

print(series)

# M = 3 # Number of stacks in the network
# K = 30  # Number of blocks in the stack
# H = 10  # Forecast horizon
# n = 5  # Look back n times the forecast horizon
# num_epochs = 5

# # Create the dataloaders
# train_loader, val_loader, test_loader = dataloaders.NBEATS_Data_Loader(series, H, n, val_size=0.2, test_size=0.1, batch_size=32)

# # Check for GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

# # Initialize model
# model = NBEATS(M, K, H, n).to(device)

# # Choose loss function and optimizer
# from custom_loss_funcs import *
# loss_func = sMAPE_loss
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Train the model
# train_NBEATS(train_loader, val_loader, model, loss_func, optimizer, device, num_epochs)