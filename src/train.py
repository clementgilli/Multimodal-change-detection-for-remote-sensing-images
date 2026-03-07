import argparse
import torch
import torch.optim as optim

from models import UNet, GradualExpansionUNet
from losses import SpectralLoss
from utils_dataset import create_dataloaders

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


def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()

    total_loss = 0
    total_mse = 0
    total_sam = 0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        pred = model(x)

        loss, mse, sam = criterion(pred, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse.item()
        total_sam += sam.item()

    n = len(loader)

    return total_loss / n, total_mse / n, total_sam / n


@torch.no_grad()
def validate(model, loader, criterion, device):

    model.eval()

    total_loss = 0
    total_mse = 0
    total_sam = 0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        loss, mse, sam = criterion(pred, y)

        total_loss += loss.item()
        total_mse += mse.item()
        total_sam += sam.item()

    n = len(loader)

    return total_loss / n, total_mse / n, total_sam / n

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

    parser.add_argument("--interpolation_mode",
                        type=str,
                        default="Bilinear",
                        choices=["ConvTranspose2d", "Bilinear"])

    parser.add_argument("--learning_mode",
                        type=str,
                        default="residual",
                        choices=["standard", "residual"])

    parser.add_argument("--lambda_sam",
                        type=float,
                        default=0.1)

    parser.add_argument("--lr",
                        type=float,
                        default=1e-4)

    parser.add_argument("--epochs",
                        type=int,
                        default=100)

    parser.add_argument("--batch_size",
                        type=int,
                        default=8)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args).to(device)

    criterion = SpectralLoss(lambda_sam=args.lambda_sam)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size
    )

    best_val_loss = float("inf")

    for epoch in range(args.epochs):

        train_loss, train_mse, train_sam = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_mse, val_sam = validate(
            model, val_loader, criterion, device
        )

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss={train_loss:.6f} MSE={train_mse:.6f} SAM={train_sam:.6f}")
        print(f"Val Loss={val_loss:.6f} MSE={val_mse:.6f} SAM={val_sam:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

    print("Training finished.")


if __name__ == "__main__":
    main()