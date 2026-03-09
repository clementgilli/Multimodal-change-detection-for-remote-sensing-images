import argparse
import torch
import torch.optim as optim

from models import UNet, GradualExpansionUNet
from losses import SpectralLoss
from utils_dataset import create_dataloaders
from tqdm import tqdm

def build_model(args):

    if args.model == "unet":
        model = UNet(
            activation=args.activation,
            interpolation_mode=args.interpolation_mode,
            learning_mode=args.learning_mode
        )

    elif args.model == "gradualexpansionunet":
        model = GradualExpansionUNet(
            activation=args.activation,
            interpolation_mode=args.interpolation_mode
        )

    else:
        raise ValueError("Unknown model")

    return model

def evalSR(HSI0,HSI_est):
    # HSI0: ground-truth HSI (cube)
    # HSI_est: estimated HSI (cube)
    HSI0 = HSI0.reshape(HSI0.shape[0],HSI0.shape[1]*HSI0.shape[2])
    HSI_est = HSI_est.reshape(HSI_est.shape[0],HSI_est.shape[1]*HSI_est.shape[2])
    
    metriqueTab = torch.zeros(HSI0.shape[1])
    for ii in range(HSI0.shape[1]):
        metriqueTab[ii] = torch.sum(HSI0[:,ii]*HSI_est[:,ii])/(torch.linalg.norm(HSI0[:,ii])*torch.linalg.norm(HSI_est[:,ii])+1e-10) # on ajoute une petite valeur pour éviter la division par zéro
        
    return torch.mean(torch.abs(metriqueTab))

@torch.no_grad()
def test(model, loader, criterion, device):

    model.eval()

    total_loss = 0
    total_mse = 0
    total_sam = 0
    total_err = 0
    total_sad_db = 0

    pbar = tqdm(loader, desc="Testing")
    for x, y in pbar:

        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        loss, mse, sam = criterion(pred, y)
        
        error = (torch.linalg.norm(y - pred) / torch.linalg.norm(y))
        sad_db = -10*torch.log10(torch.abs(1-(evalSR(y, pred))))

        total_loss += loss.item()
        total_mse += mse.item()
        total_sam += sam.item()
        total_err += error.item()
        total_sad_db += sad_db.item()
        
        pbar.set_postfix(Loss=f"{loss.item():.4f}", MSE=f"{mse.item():.4f}", SAM=f"{sam.item():.4f}")

    n = len(loader)

    return total_loss / n, total_mse / n, total_sam / n, total_err / n, total_sad_db / n

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model",
                        type=str,
                        default="unet",
                        choices=["unet", "gradualexpansionunet"])

    parser.add_argument("--activation",
                        type=str,
                        default="silu",
                        choices=["relu", "leakyrelu", "silu"])
    
    parser.add_argument("--learning_mode",
                        type=str,
                        default="residual",
                        choices=["standard", "residual"])

    parser.add_argument("--interpolation_mode",
                        type=str,
                        default="Bilinear",
                        choices=["ConvTranspose2d", "Bilinear"])

    parser.add_argument("--lambda_sam",
                        type=float,
                        default=0.1)

    parser.add_argument("--batch_size",
                        type=int,
                        default=8)
    
    parser.add_argument("--num_workers",
                        type=int,
                        default=4)
    
    parser.add_argument("--load_model",
                        type=str,
                        default=None)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args).to(device)
    
    if args.load_model is not None:
        model.load_state_dict(torch.load(args.load_model, map_location=device))
        print(f"Model loaded from {args.load_model}")
    else:
        print(f"No model loaded")

    criterion = SpectralLoss(lambda_sam=args.lambda_sam).to(device)

    _, _, test_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    

    test_loss, test_mse, test_sam, test_err, test_sad_db = test(
        model, test_loader, criterion, device
    )

    print(f"Test Loss={test_loss:.6f} MSE={test_mse:.6f} SAM={test_sam:.6f} \nERR={test_err:.6f} %, SAD={test_sad_db:.6f} dB")


if __name__ == "__main__":
    main()
