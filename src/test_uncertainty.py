import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models import UNet
from utils_dataset import create_dataloaders, UncertaintyDataset
from tqdm import tqdm

@torch.no_grad()
def test_uncertainty(model, loader, device, num_channels=230):
    model.eval()

    total_mse_per_channel = torch.zeros(num_channels, device=device)
    total_mae_per_channel = torch.zeros(num_channels, device=device)

    pbar = tqdm(loader, desc="Testing Uncertainty per Channel")
    
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        pred_uncertainty = model(x)

        # [B, 230, H, W]
        mse_raw = F.mse_loss(pred_uncertainty, y, reduction='none')
        mae_raw = F.l1_loss(pred_uncertainty, y, reduction='none')

        mse_per_channel = mse_raw.mean(dim=(0, 2, 3))
        mae_per_channel = mae_raw.mean(dim=(0, 2, 3))

        total_mse_per_channel += mse_per_channel
        total_mae_per_channel += mae_per_channel
        
        pbar.set_postfix(Global_MAE=f"{mae_per_channel.mean().item():.4f}")

    n = len(loader)

    final_mse = (total_mse_per_channel / n).cpu().numpy()
    final_mae = (total_mae_per_channel / n).cpu().numpy()

    return final_mse, final_mae

def plot_and_save_errors(mae_per_channel, mse_per_channel):
    channels = np.arange(len(mae_per_channel))
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(channels, mae_per_channel, label='MAE', color='blue', linewidth=2)
    plt.plot(channels, mse_per_channel, label='MSE', color='red', linestyle='--', alpha=0.7)
    
    plt.title("Uncertainty Prediction Error Per Channel", fontsize=14)
    plt.xlabel("Channel", fontsize=12)
    plt.ylabel("Error", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plot_path = "data/uncertainty_error_per_channel.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\n Error plot saved in: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Test Uncertainty Prediction Model")

    parser.add_argument("--activation",
                        type=str,
                        default="silu",
                        choices=["relu", "leakyrelu", "silu"])

    parser.add_argument("--interpolation_mode",
                        type=str,
                        default="Bilinear",
                        choices=["ConvTranspose2d", "Bilinear"])

    parser.add_argument("--batch_size",
                        type=int,
                        default=8)
    
    parser.add_argument("--num_workers",
                        type=int,
                        default=4)
    
    parser.add_argument("--load_model",
                        type=str,
                        required=True, 
                        help="Path to the trained uncertainty model (.pth)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(
        in_channels=242, 
        activation=args.activation,
        interpolation_mode=args.interpolation_mode,
        learning_mode="standard"
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(args.load_model, map_location=device))
        print(f"Model loaded from  {args.load_model}")
    except Exception as e:
        print(f"Couldn't load model: {e}")
        return

    _, _, test_loader = create_dataloaders(
        dataset_class=UncertaintyDataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    test_mse_ch, test_mae_ch = test_uncertainty(model, test_loader, device)

    print("\n" + "="*50)
    print(f"Global Test MSE : {test_mse_ch.mean():.6f}")
    print(f"Global Test MAE : {test_mae_ch.mean():.6f}")
    print(f"Channel with highest MAE : Channel {np.argmax(test_mae_ch)} (MAE={np.max(test_mae_ch):.6f})")
    print(f"Channel with lowest MAE : Channel {np.argmin(test_mae_ch)} (MAE={np.min(test_mae_ch):.6f})")
    print(f"Channel with highest MSE : Channel {np.argmax(test_mse_ch)} (MSE={np.max(test_mse_ch):.6f})")
    print(f"Channel with lowest MSE : Channel {np.argmin(test_mse_ch)} (MSE={np.min(test_mse_ch):.6f})")
    
    plot_and_save_errors(test_mae_ch, test_mse_ch)

if __name__ == "__main__":
    main()