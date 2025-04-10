import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
import torch.nn.functional as F

class KITTIDataset(Dataset):
    def __init__(self, root_dir, image_transform=None, disparity_transform=None):
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.disparity_transform = disparity_transform

        # Get the base names of disparity map files
        disparity_files = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root_dir, 'disp_occ_0'))]

        # Build lists of file paths
        self.left_images = [os.path.join(root_dir, 'image_2', f"{scene}.png") for scene in disparity_files]
        self.right_images = [os.path.join(root_dir, 'image_3', f"{scene}.png") for scene in disparity_files]
        self.disparity_maps = [os.path.join(root_dir, 'disp_occ_0', f"{scene}.png") for scene in disparity_files]

        # Check file existence
        for left, right, disp in zip(self.left_images, self.right_images, self.disparity_maps):
            if not (os.path.exists(left) and os.path.exists(right) and os.path.exists(disp)):
                raise FileNotFoundError(f"Missing file: {left}, {right}, or {disp}")

        print(f"Loaded {len(self.disparity_maps)} scenes with disparity maps.")

    def __len__(self):
        return len(self.disparity_maps)

    def __getitem__(self, idx):
        left_img = Image.open(self.left_images[idx]).convert('RGB')
        right_img = Image.open(self.right_images[idx]).convert('RGB')
        disparity = Image.open(self.disparity_maps[idx]).convert('L')

        if self.image_transform:
            left_img = self.image_transform(left_img)
            right_img = self.image_transform(right_img)
        if self.disparity_transform:
            disparity = self.disparity_transform(disparity)
            disparity = disparity * 256.0  # Scale disparity values if needed

        stereo_pair = torch.cat((left_img, right_img), dim=0)
        return stereo_pair, disparity


class DrivingStereoDataset(Dataset):
    def __init__(self, root_dir, image_transform=None, depth_transform=None):
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.depth_transform = depth_transform  # Renamed for clarity

        # List all files in the 'left' directory
        scene_list = os.listdir(os.path.join(root_dir, 'left'))

        # Construct file paths
        self.left_images = [os.path.join(root_dir, 'left', f) for f in scene_list]
        self.right_images = [os.path.join(root_dir, 'right', f) for f in scene_list]
        # Load depth maps from 'depth' instead of disparity maps from 'diff'
        self.depth_maps = [os.path.join(root_dir, 'depth', os.path.splitext(f)[0] + '.png') for f in scene_list]

        # Verify file existence to catch errors early
        for left, right, depth in zip(self.left_images, self.right_images, self.depth_maps):
            if not (os.path.exists(left) and os.path.exists(right) and os.path.exists(depth)):
                raise FileNotFoundError(f"Missing file: {left}, {right}, or {depth}")

    def __len__(self):
        return len(self.left_images)

    import numpy as np
    import torch.nn.functional as F

    def __getitem__(self, idx):
        # Load stereo images
        left_img = Image.open(self.left_images[idx]).convert('RGB')
        right_img = Image.open(self.right_images[idx]).convert('RGB')

        # Load depth map (uint16)
        depth = Image.open(self.depth_maps[idx])
        # Convert to numpy array and scale
        depth_array = np.array(depth, dtype=np.float32)  # uint16 -> float32
        depth_array = depth_array / 256.0  # Convert to meters (adjust scaling as needed)

        # Convert to tensor
        depth_tensor = torch.from_numpy(depth_array).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        # Resize to match model output
        depth_tensor = F.interpolate(depth_tensor, size=(256, 512), mode='bilinear', align_corners=False)
        depth = depth_tensor.squeeze(0)  # [1, 256, 512]

        # Apply image transforms
        if self.image_transform:
            left_img = self.image_transform(left_img)
            right_img = self.image_transform(right_img)

        stereo_pair = torch.cat((left_img, right_img), dim=0)
        return stereo_pair, depth

