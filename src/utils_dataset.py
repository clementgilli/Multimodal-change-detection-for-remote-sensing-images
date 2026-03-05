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

def prepare_dataset_offline(file_paths, patch_size=256, srf_path=DEFAULT_SRF_PATH):
    srf_matrix = np.load(srf_path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    for file in tqdm(file_paths):
        base_name = Path(file).name.replace('-prs.nc', '')
        x_path = CACHE_DIR / f'{base_name}_X_patches.npy'
        y_path = CACHE_DIR / f'{base_name}_y_patches.npy'
        
        if x_path.exists() and y_path.exists():
            continue
            
        with xr.open_dataset(file) as ds:
            img_hyper = ds["sr"].to_numpy().astype(np.float32)
            
        h, w, c_hyper = img_hyper.shape
        c_multi = srf_matrix.shape[1]
           
        img_multi = from_hyper_to_multi(img_hyper, srf_matrix)
        
        h_crop = h - (h % patch_size)
        w_crop = w - (w % patch_size)
        n_patches_h = h_crop // patch_size
        n_patches_w = w_crop // patch_size
        total_patches = n_patches_h * n_patches_w
            
        X_mmap = np.lib.format.open_memmap(
            str(y_path), mode='w+', dtype=np.float32, 
            shape=(total_patches, c_multi, patch_size, patch_size)
        )
        
        y_mmap = np.lib.format.open_memmap(
            str(x_path), mode='w+', dtype=np.float32, 
            shape=(total_patches, c_hyper, patch_size, patch_size)
        )
        
        patch_idx = 0
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                r = i * patch_size
                c = j * patch_size
                
                patch_hyper = img_hyper[r:r+patch_size, c:c+patch_size, :]
                patch_multi = img_multi[r:r+patch_size, c:c+patch_size, :]
                
                X_mmap[patch_idx] = np.transpose(patch_multi, (2, 0, 1))
                y_mmap[patch_idx] = np.transpose(patch_hyper, (2, 0, 1))
                
                patch_idx += 1
                
        del X_mmap, y_mmap, img_hyper, img_multi
        
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