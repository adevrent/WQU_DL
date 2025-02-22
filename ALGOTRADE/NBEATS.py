# Import necessary libraries
from matplotlib import pyplot as plt
import torch
from torch import nn

import yfinance as yf
from datetime import date

# Generic architecture
class NBEATSBlock(nn.Module):
    def __init__(self, H, n, dropout_prob=0.1):
        """
        Parameters:
        H (int): Forecast horizon (number of future time steps to predict)
        n (int): How many times the forecast horizon to look back
        """
        super().__init__()
        self.stem = nn.Sequential(  # input size: (batch_size, 1, H*n)
            nn.Linear(in_features=H*n, out_features=512), nn.ReLU(), nn.Dropout(dropout_prob),
            nn.Linear(in_features=512, out_features=512), nn.ReLU(), nn.Dropout(dropout_prob),
            nn.Linear(in_features=512, out_features=512), nn.ReLU(), nn.Dropout(dropout_prob),
            nn.Linear(in_features=512, out_features=512), nn.ReLU()
        )

        # Generate thetas for backcast and forecast
        self.b1 = nn.Linear(in_features=512, out_features=512)
        self.b2 = nn.Linear(in_features=512, out_features=512)

        # Generate the backcast and forecast
        self.g_b = nn.Linear(in_features=512, out_features=H*n)
        self.g_f = nn.Linear(in_features=512, out_features=H)

    def forward(self, x):
        x = self.stem(x)

        # Generate thetas for backcast and forecast
        theta_b = self.b1(x)  # backcast
        theta_f = self.b2(x)  # forecast

        x_bc = self.g_b(theta_b)
        x_fc = self.g_f(theta_f)

        return x_bc, x_fc
    
class NBEATSStack(nn.Module):
    def __init__(self, K, H, n):
        """
        Parameters:
        K (int): Number of blocks in the stack
        H (int): Forecast horizon (number of future time steps to predict)
        n (int): How many times the forecast horizon to look back
        """
        self.H = H
        super().__init__()
        self.blocks = nn.ModuleList([NBEATSBlock(H, n) for _ in range(K)])

    def forward(self, x):
        x_fc_sum = torch.zeros(x.size(0), self.H, dtype=x.dtype).to(x.device)
        residual = x  # start with the original input
        for block in self.blocks:
            x_bc, x_fc = block(residual)
            residual = residual - x_bc  # update residual using the block’s input
            x_fc_sum += x_fc
        return residual, x_fc_sum
    
class NBEATS(nn.Module):
    def __init__(self, M, K, H, n):
        """
        Parameters:
        M (int): Number of stacks in the network
        K (int): Number of blocks in the stack
        H (int): Forecast horizon (number of future time steps to predict)
        n (int): How many times the forecast horizon to look back
        """
        super().__init__()
        self.H = H
        self.stacks = nn.ModuleList([NBEATSStack(K, H, n) for _ in range(M)])
    
    def forward(self, x):
        # Initialize forecast accumulator
        forecast_total = torch.zeros(x.size(0), self.H, 
                                  device=x.device, dtype=x.dtype)
        residual = x
        
        for stack in self.stacks:
            # Process through stack
            residual, stack_forecast = stack(residual)
            # Aggregate forecasts
            forecast_total += stack_forecast
            
        return forecast_total

def train_NBEATS(train_loader, val_loader, model, loss_func, optimizer, device, num_epochs, feature="prices"):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10
    epochs_without_improve = 0

    if feature == "prices":
        save_path_best = "best_nbeats_price_model.pth"
        save_path_final = "nbeats_price_model_weights_final.pth"
    else:
        save_path_best = "best_nbeats_logr_model.pth"
        save_path_final = "nbeats_logr_model_weights_final.pth"

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        running_train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)   # shape: (batch_size, H*n)
            targets = targets.to(device)   # shape: (batch_size, H)
            
            optimizer.zero_grad()
            forecasts = model(inputs)      # forecast shape: (batch_size, H)
            loss = loss_func(forecasts, targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation loop
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                forecasts = model(inputs)
                loss = loss_func(forecasts, targets)
                running_val_loss += loss.item() * inputs.size(0)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f} - Val Loss: {epoch_val_loss:.4f}")
        
        # Early stopping: check if validation loss improved
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_without_improve = 0
            # Optionally save the best model here
            torch.save(model.state_dict(), save_path_best)
            print(f"    Best model weights so far saved to {save_path_best}")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Save the final model weights
    torch.save(model.state_dict(), save_path_final)
    print(f"    Final model weights saved to {save_path_final}")

    # Plot the training and validation loss curves
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axs[0].plot(train_losses, marker='o', label="Training Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("MSE Loss")
    axs[0].set_title("Training Loss Curve")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(val_losses, marker='x', color='orange', label="Validation Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("MSE Loss")
    axs[1].set_title("Validation Loss Curve")
    axs[1].legend()
    axs[1].grid()

    fig.tight_layout()
    plt.show(block=True)