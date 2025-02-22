import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch import nn

import dataloaders
from NBEATS import *
from custom_loss_funcs import *

data_filepath = r"C:\Users\Atakan\atakan_python\WQU_DL\ALGOTRADE\static_files\btc_1min.csv"
start_date = "2019-01-01"
end_date = "2021-03-01"

# Read the CSV file
series = dataloaders.read_csv(data_filepath, date_col="Timestamp", value_col="Close", start_date=start_date, end_date=end_date)
print(series.head())

# Load hyperparameters
M, K, H, n, batch_size, num_epochs = dataloaders.load_hyperparams()

# Create the dataloaders
train_loader, val_loader, test_loader = dataloaders.NBEATS_Data_Loader(series, H, n, val_size=0.2, test_size=0.1, batch_size=batch_size)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize model
model = NBEATS(M, K, H, n).to(device)

# Call test function
weights_filepath = "best_nbeats_price_model.pth"
loss_func = sMAPE_loss
test_NBEATS(test_loader, model, loss_func, device, weights_filepath)