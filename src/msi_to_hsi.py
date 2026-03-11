import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from models import UNet, GradualExpansionUNet

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

def generate_hsi_simulated(model, cache_dir, device, batch_size=8):
    model.eval()
    cache_path = Path(cache_dir)
    msi_files = sorted(cache_path.glob("*_MSI_simulated.npy"))
    
    with torch.no_grad():
        for msi_file in tqdm(msi_files, desc="Génération HSI Simulée"):
            hsi_sim_file = msi_file.with_name(msi_file.name.replace("MSI_simulated", "HSI_simulated"))
            
            if hsi_sim_file.exists():
                continue
                
            msi_data = np.load(msi_file) 
            num_patches, c_msi, h, w = msi_data.shape
            
            dummy_input = torch.from_numpy(msi_data[0:1]).to(device)
            c_hsi = model(dummy_input).shape[1]
            
            hsi_sim_data = np.zeros((num_patches, c_hsi, h, w), dtype=np.float32)
            
            for i in range(0, num_patches, batch_size):
                batch_msi = torch.from_numpy(msi_data[i:i+batch_size]).to(device)
                batch_msi = torch.clamp(batch_msi, min=0.0, max=1.0)
                
                batch_hsi_sim = model(batch_msi)
                
                hsi_sim_data[i:i+batch_size] = batch_hsi_sim.cpu().numpy()
                
            np.save(hsi_sim_file, hsi_sim_data)
                        
            del msi_data, hsi_sim_data
            
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        type=str,
                        required=True,
                        help="Path to the trained model file (e.g., .pth)")
    
    parser.add_argument("--cache_dir",
                        type=str,
                        required=True,
                        help="Directory where the simulated MSI files are stored and where the generated HSI files will be saved")
    
    parser.add_argument("--batch_size",
                        type=int,
                        default=8,
                        help="Batch size for generation")
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

    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Model loaded from {args.model_path}")
    
    generate_hsi_simulated(model, args.cache_dir, device, args.batch_size)