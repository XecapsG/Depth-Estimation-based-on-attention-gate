import torch

def evaluate_model(model, val_loader, device='cpu'):
    model.eval()
    total_epe = 0.0
    with torch.no_grad():
        for stereo_pair, disparity in val_loader:
            stereo_pair = stereo_pair.to(device)
            disparity = disparity.to(device)
            outputs = model(stereo_pair)
            epe = torch.mean(torch.abs(outputs - disparity))
            total_epe += epe.item()
    avg_epe = total_epe / len(val_loader)
    print(f'Average End-Point Error: {avg_epe:.4f}')

