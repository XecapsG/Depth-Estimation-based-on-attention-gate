import os
import torch
import matplotlib.pyplot as plt

import os
import torch
import torch.nn.functional as F


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu',
                save_path='weights/model_best.pth'):
    train_maes = []  # Store training MAE (same as train_losses)
    val_maes = []  # Store validation MAE (same as val_losses)
    val_rmses = []  # Store validation RMSE
    best_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for stereo_pair, depth in train_loader:
            stereo_pair = stereo_pair.to(device)
            depth = depth.to(device)
            optimizer.zero_grad()
            outputs = model(stereo_pair)
            loss = criterion(outputs, depth)  # MAE
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_mae = running_loss / len(train_loader)
        train_maes.append(avg_train_mae)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training MAE: {avg_train_mae:.4f}')

        # Validation phase
        model.eval()
        val_mae = 0.0
        val_mse = 0.0
        with torch.no_grad():
            for stereo_pair, depth in val_loader:
                stereo_pair = stereo_pair.to(device)
                depth = depth.to(device)
                outputs = model(stereo_pair)
                mae = criterion(outputs, depth).item()  # MAE per batch
                mse = F.mse_loss(outputs, depth).item()  # MSE per batch
                val_mae += mae
                val_mse += mse
        avg_val_mae = val_mae / len(val_loader)
        avg_val_mse = val_mse / len(val_loader)
        val_rmse = avg_val_mse ** 0.5  # RMSE = sqrt(MSE)
        val_maes.append(avg_val_mae)
        val_rmses.append(val_rmse)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation MAE: {avg_val_mae:.4f}')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation SMAE: {val_rmse:.4f}')

        # Save the model if validation MAE improves
        if avg_val_mae < best_loss:
            best_loss = avg_val_mae
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            # print(f"Best model saved to {save_path}")

    return train_maes, val_maes, val_rmses


def plot_losses(train_losses, val_losses):
    """
    Plot training and validation losses over epochs.

    Args:
        train_losses (list): List of training losses for each epoch
        val_losses (list): List of validation losses for each epoch
    """
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Changes Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_metrics(train_maes, val_maes, val_rmses):
    epochs = range(1, len(train_maes) + 1)
    plt.plot(epochs, train_maes, label='Training MAE')
    plt.plot(epochs, val_maes, label='Validation MAE')
    # plt.plot(epochs, val_rmses, label='Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE Loss')
    plt.title('Training & Validation MAE Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
