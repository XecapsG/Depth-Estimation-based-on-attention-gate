import cv2
import torch
import numpy as np
from dataset import DrivingStereoDataset
from model import SimpleUNetWithAttention
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

def generate_depth_video(model, val_loader, device, output_path="depth_video.mp4"):
    model.eval()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = None

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(val_loader):  # 忽略真实的深度图
            images = images.to(device)  # [B, 6, H, W]，包含左右图像
            outputs = model(images)  # [B, 1, H, W]，预测的深度图

            # 将数据移到 CPU 并转换为 numpy
            outputs = outputs.cpu().numpy()  # [B, 1, H, W]
            images = images.cpu()  # [B, 6, H, W]

            for i in range(outputs.shape[0]):
                # 提取左图像 [H, W, 3]
                left_img = images[i, :3].permute(1, 2, 0).numpy()  # [H, W, 3]
                left_img = (left_img * 255).astype(np.uint8)  # 转换为 0-255 的 uint8 格式

                # 提取预测深度图 [H, W]
                depth_map = outputs[i, 0]  # [H, W]
                # 归一化到 0-255 范围
                depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6) * 255
                depth_map = depth_map.astype(np.uint8)
                # 应用彩色映射
                depth_map_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

                # 水平拼接左图像和彩色深度图
                combined_img = np.hstack((left_img, depth_map_colored))

                # 初始化视频写入器（仅在第一帧时）
                if video_writer is None:
                    height, width, _ = combined_img.shape
                    video_writer = cv2.VideoWriter(output_path, fourcc, 10.0, (width, height), isColor=True)
                    print(f"视频写入器已初始化：({width}, {height})")

                # 写入当前帧
                video_writer.write(combined_img)

    if video_writer is not None:
        video_writer.release()
    print(f"深度估计视频已保存至 {output_path}")

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleUNetWithAttention().to(device)
weights_path = 'weights/model_best.pth'

# 加载模型权重
model.load_state_dict(torch.load(weights_path, map_location=device))
print(f"Loaded model weights from {weights_path}")

# 数据预处理
image_transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor()
])
depth_transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor()
])

# 加载数据集
root_dir = "Driving_stereo/train"
dataset = DrivingStereoDataset(root_dir, image_transform=image_transform, depth_transform=depth_transform)
val_size = int(0.2 * len(dataset))
val_indices = list(range(val_size))
val_dataset = Subset(dataset, val_indices)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 生成视频
output_video_path = "output_depth_video.mp4"
generate_depth_video(model, val_loader, device, output_video_path)