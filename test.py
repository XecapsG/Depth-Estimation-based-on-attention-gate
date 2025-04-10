import torch
from torchvision import transforms
from dataset import DrivingStereoDataset
from model import SimpleUNetWithAttention
from torch.utils.data import random_split
import matplotlib.pyplot as plt


def predict_depth(model, dataset, index, device):
    model.eval()
    stereo_pair, ground_truth_depth = dataset[index]  # stereo_pair: [6, H, W]
    left_img = stereo_pair[:3].permute(1, 2, 0).numpy()  # Extract left image: [H, W, 3]
    stereo_pair = stereo_pair.unsqueeze(0).to(device)  # Prepare for model: [1, 6, H, W]
    with torch.no_grad():
        predicted_depth = model(stereo_pair).squeeze().cpu().numpy()  # [H, W]
    ground_truth_depth = ground_truth_depth.squeeze().numpy()  # [H, W]
    return left_img, predicted_depth, ground_truth_depth


def visualize_depth(left_img, predicted_depth, ground_truth_depth, save_path_prefix=None):
    # Plot Left Image
    plt.figure(figsize=(6, 6))
    plt.imshow(left_img)
    plt.title('Left Image')
    plt.axis('off')
    if save_path_prefix:
        plt.savefig(f'{save_path_prefix}_left_image.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Plot Ground Truth Depth
    plt.figure(figsize=(6, 6))
    plt.imshow(ground_truth_depth, cmap='jet')
    plt.title('Ground Truth Depth')
    plt.axis('off')
    if save_path_prefix:
        plt.savefig(f'{save_path_prefix}_ground_truth_depth.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Plot Predicted Depth
    plt.figure(figsize=(6, 6))
    plt.imshow(predicted_depth, cmap='jet')
    plt.title('Predicted Depth')
    plt.axis('off')
    if save_path_prefix:
        plt.savefig(f'{save_path_prefix}_predicted_depth.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = SimpleUNetWithAttention().to(device)
    weights_path = 'weights/model_best.pth'
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded model weights from {weights_path}")
    except FileNotFoundError:
        print(f"Error: Model weights not found at {weights_path}. Please train the model first.")
        exit()

    image_transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor()
    ])
    depth_transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor()
    ])

    root_dir = "Driving_stereo/train"
    try:
        full_dataset = DrivingStereoDataset(root_dir, image_transform=image_transform, depth_transform=depth_transform)
        print(f"Dataset loaded with {len(full_dataset)} samples.")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check the dataset path and structure.")
        exit()

    # Split the dataset into training (80%) and validation (20%) subsets
    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Validation set has {len(val_dataset)} samples.")

    # Choose an index within the validation set (e.g., the first image)
    index = 0
    left_img, predicted_depth, ground_truth_depth = predict_depth(model, val_dataset, index, device)

    # Visualize the images separately and save them
    visualize_depth(left_img, predicted_depth, ground_truth_depth, save_path_prefix='depth_visualization')

    # Uncomment to print depth ranges if needed
    # print("Ground Truth Depth Range:", ground_truth_depth.min(), ground_truth_depth.max())
    # print("Predicted Depth Range:", predicted_depth.min(), predicted_depth.max())