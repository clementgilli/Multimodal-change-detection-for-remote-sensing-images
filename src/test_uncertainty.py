import argparse
import torch
import torch.nn.functional as F

from models import UNet
from utils_dataset import create_dataloaders, UncertaintyDataset
from tqdm import tqdm

@torch.no_grad()
def test_uncertainty(model, loader, device):
    model.eval()

    total_mse = 0
    total_mae = 0

    pbar = tqdm(loader, desc="Testing Uncertainty")
    
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device) 

        pred_uncertainty = model(x)

        mse = F.mse_loss(pred_uncertainty, y)
        mae = F.l1_loss(pred_uncertainty, y)

        total_mse += mse.item()
        total_mae += mae.item()
        
        pbar.set_postfix(MSE=f"{mse.item():.6f}", MAE=f"{mae.item():.6f}")

    n = len(loader)

    return total_mse / n, total_mae / n

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

    test_mse, test_mae = test_uncertainty(model, test_loader, device)

    print("\n" + "="*50)
    print(f"Test MSE : {test_mse:.6f} (Erreur Quadratique Moyenne)")
    print(f"Test MAE : {test_mae:.6f} (Erreur Absolue Moyenne)")

if __name__ == "__main__":
    main()