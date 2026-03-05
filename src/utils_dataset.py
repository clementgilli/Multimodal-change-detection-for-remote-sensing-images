import os
import glob
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
from einops import rearrange
from tqdm import tqdm
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()

# utils_dataset.py is in "project/src/", .parent.parent links to "project/"
ROOT_DIR = CURRENT_FILE.parent.parent

DATA_DIR = ROOT_DIR / 'data'
CACHE_DIR = DATA_DIR / 'patches_cache'
DEFAULT_SRF_PATH = DATA_DIR / 'srf_matrix_norm_s2b.npy'

def from_hyper_to_multi(img_hyper, srf_matrix_norm):
    h, w, c = img_hyper.shape
    hyper_2d = img_hyper.reshape(h * w, c)              
    multi_2d = np.dot(hyper_2d, srf_matrix_norm)         
    image_multispec = multi_2d.reshape(h, w, srf_matrix_norm.shape[1])
    return image_multispec

def prepare_dataset_offline(file_paths, patch_size=64, srf_path=DEFAULT_SRF_PATH):
    srf_matrix = np.load(srf_path)
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    for file in tqdm(file_paths):
        base_name = Path(file).name.replace('-prs.nc', '')
        
        x_path = CACHE_DIR / f'{base_name}_X_patches.npy'
        y_path = CACHE_DIR / f'{base_name}_y_patches.npy'
        
        if x_path.exists() and y_path.exists():
            continue
            
        with xr.open_dataset(file) as ds:
            img_hyper = ds["sr"].to_numpy()
            
        h, w, c = img_hyper.shape
        img_multi = from_hyper_to_multi(img_hyper, srf_matrix)
        
        h_crop = h - (h % patch_size)
        w_crop = w - (w % patch_size)
        img_hyper = img_hyper[:h_crop, :w_crop, :]
        img_multi = img_multi[:h_crop, :w_crop, :]
        
        X_patches = rearrange(img_multi, '(h p1) (w p2) c -> (h w) c p1 p2', p1=patch_size, p2=patch_size)
        y_patches = rearrange(img_hyper, '(h p1) (w p2) c -> (h w) c p1 p2', p1=patch_size, p2=patch_size)
        
        np.save(x_path, X_patches.astype(np.float32))
        np.save(y_path, y_patches.astype(np.float32))
        
class SpectralDataset(Dataset):
    def __init__(self, cache_dir=CACHE_DIR):

        self.x_files = sorted(glob.glob(str(cache_dir / "*_X_patches.npy")))
        self.y_files = sorted(glob.glob(str(cache_dir / "*_y_patches.npy")))
        
        assert len(self.x_files) > 0, f"No patch found in {cache_dir}. Run prepare_dataset_offline() first."
        assert len(self.x_files) == len(self.y_files), "Mismatch between X and y files."
        
        self.index_map = []
        self.mmap_x_arrays = []
        self.mmap_y_arrays = []
        
        for file_idx, (x_path, y_path) in enumerate(zip(self.x_files, self.y_files)):
            x_mmap = np.load(x_path, mmap_mode='r')
            y_mmap = np.load(y_path, mmap_mode='r')
            
            self.mmap_x_arrays.append(x_mmap)
            self.mmap_y_arrays.append(y_mmap)
            
            num_patches = x_mmap.shape[0]
            
            for patch_idx in range(num_patches):
                self.index_map.append((file_idx, patch_idx))
                
        print(f"Loaded {len(self.index_map)} patches from {len(self.x_files)} files.")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, patch_idx = self.index_map[idx]
        
        x_patch = self.mmap_x_arrays[file_idx][patch_idx].copy()
        y_patch = self.mmap_y_arrays[file_idx][patch_idx].copy()
        
        return torch.from_numpy(x_patch), torch.from_numpy(y_patch)