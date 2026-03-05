from utils_dataset import *
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Prepare dataset by converting hyperspectral images to multispectral and creating patches.")
    parser.add_argument("--patch_size", type=int, default=256, help="Size of the patches to create")
    
    args = parser.parse_args()
    PATCH_SIZE = args.patch_size
    
    file_paths = glob.glob(str(DATA_DIR / "*" / "*-prs.nc"))
    prepare_dataset_offline(file_paths, patch_size=PATCH_SIZE)