# Import necessary libraries
from matplotlib import pyplot as plt
import torch
from torch import nn
import numpy as np

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

def train_NBEATS(train_loader, val_loader, model, loss_func, optimizer, device, num_epochs, feature="prices", patience=10):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
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

def test_NBEATS(test_loader, model, loss_func, device, weights_filepath):
    """
    Loads the optimum weights, evaluates the model on the test set, prints the sMAPE value,
    and plots forecast vs. true curves for 10 random samples (5x2 subplots). 
    Additionally, shows ratio difference (%) on a secondary y-axis.
    
    Parameters:
        test_loader (DataLoader): DataLoader for the test set.
        model (nn.Module): The NBEATS model instance.
        loss_func (callable): The loss function to compute sMAPE.
        device (torch.device): Device (cuda or cpu) on which to run the evaluation.
        weights_filepath (str): Path to the optimum weights file (.pth).
    """
    # Load the optimum weights
    model.load_state_dict(torch.load(weights_filepath, map_location=device))
    model.to(device)
    model.eval()
    
    forecasts_list = []
    targets_list = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            forecasts = model(inputs)
            forecasts_list.append(forecasts)
            targets_list.append(targets)
    
    # Concatenate all batches
    forecasts_all = torch.cat(forecasts_list, dim=0)  # shape: (num_samples, H)
    targets_all = torch.cat(targets_list, dim=0)      # shape: (num_samples, H)
    
    # Compute sMAPE on the entire test set
    test_smape = loss_func(forecasts_all, targets_all).item()
    print(f"Test sMAPE: {test_smape:.4f}")
    
    # Plot 10 random samples in a single figure with 5 rows, 2 columns
    num_samples = forecasts_all.shape[0]
    sample_indices = np.random.choice(num_samples, size=10, replace=False)
    
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12, 15), sharex=False)
    axes = axes.flatten()  # so we can iterate easily

    for i, idx in enumerate(sample_indices):
        ax = axes[i]
        forecast_sample = forecasts_all[idx].cpu().numpy()
        target_sample = targets_all[idx].cpu().numpy()
        
        # Plot forecast vs. actual
        ax.plot(target_sample, marker='o', label="Actual")
        ax.plot(forecast_sample, marker='x', label="Forecast")
        ax.set_title(f"Sample {idx} - Forecast vs Actual", fontsize=10)
        ax.set_ylabel("Value")
        ax.grid(True)
        
        # Only label the x-axis for the bottom row (the last 2 plots)
        if i < 8:  
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Time Step")

        # Create a twin axis for ratio difference
        ax2 = ax.twinx()
        # ratio_diff = 100 * (Forecast/Actual - 1)
        # Avoid division by zero by adding a small epsilon if necessary
        eps = 1e-8
        ratio_diff = 100.0 * ((forecast_sample + eps) / (target_sample + eps) - 1.0)
        ax2.plot(ratio_diff, color='tab:red', linestyle='--', label="Ratio Diff (%)")
        ax2.set_ylabel("Ratio Diff (%)", color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        # Combine legends from both axes
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    plt.subplots_adjust(hspace=0.4, wspace=0.3)  # increase spacing
    plt.tight_layout()
    plt.show(block=True)