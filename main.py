import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import DrivingStereoDataset
from model import SimpleUNetWithAttention
from train import train_model, plot_metrics  # Update import
from eval import evaluate_model
from torch.utils.data import Subset
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define transformations
    image_transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor()
    ])
    depth_transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor()
    ])

    # Set the root directory
    root_dir = "Driving_stereo\\train"
    dataset = DrivingStereoDataset(root_dir, image_transform=image_transform, depth_transform=depth_transform)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    # train_dataset = Subset(dataset, range(0, train_size))
    #
    # # The last 20% for validation
    # val_dataset = Subset(dataset, range(train_size, len(dataset)))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Initialize model, loss, and optimizer
    model = SimpleUNetWithAttention().to(device)
    model.apply(init_weights)
    criterion = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # Define save path
    save_path = 'weights/model_best.pth'

    # Train the model and get metrics
    train_maes, val_maes, val_rmses = train_model(model, train_loader, val_loader, criterion, optimizer,
                                                  num_epochs=20, device=device, save_path=save_path)

    # Plot the metrics
    plot_metrics(train_maes, val_maes, val_rmses)

    # Load the best model and evaluate
    model.load_state_dict(torch.load(save_path))
    model.eval()
    evaluate_model(model, val_loader, device=device)