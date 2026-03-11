import os
import glob
import random
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader, Subset
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
    c_multi = srf_matrix.shape[1]
    
    for file in tqdm(file_paths):
        base_name = Path(file).name.replace('-prs.nc', '')
        x_path = CACHE_DIR / f'{base_name}_X_patches.npy'
        y_path = CACHE_DIR / f'{base_name}_y_patches.npy'
        
        if x_path.exists() and y_path.exists():
            continue
            
        with xr.open_dataset(file) as ds:
            h, w, c_hyper = ds["sr"].shape
            
            h_crop = h - (h % patch_size)
            w_crop = w - (w % patch_size)
            n_patches_h = h_crop // patch_size
            n_patches_w = w_crop // patch_size
            total_patches = n_patches_h * n_patches_w
            
            X_mmap = np.lib.format.open_memmap(
                str(x_path), mode='w+', dtype=np.float32, 
                shape=(total_patches, c_multi, patch_size, patch_size)
            )
            y_mmap = np.lib.format.open_memmap(
                str(y_path), mode='w+', dtype=np.float32, 
                shape=(total_patches, c_hyper, patch_size, patch_size)
            )
            
            patch_idx = 0
            for i in range(n_patches_h):
                for j in range(n_patches_w):
                    r = i * patch_size
                    c = j * patch_size
                    
                    patch_hyper = ds["sr"][r:r+patch_size, c:c+patch_size, :].to_numpy().astype(np.float32)
                    
                    hyper_2d = patch_hyper.reshape(-1, c_hyper)              
                    multi_2d = np.dot(hyper_2d, srf_matrix)         
                    patch_multi = multi_2d.reshape(patch_size, patch_size, c_multi).astype(np.float32)
                    
                    X_mmap[patch_idx] = np.transpose(patch_multi, (2, 0, 1)) 
                    y_mmap[patch_idx] = np.transpose(patch_hyper, (2, 0, 1))
                    
                    patch_idx += 1
            
            X_mmap.flush()
            y_mmap.flush()
            del X_mmap, y_mmap
        
class SpectralDataset(Dataset):
    def __init__(self, cache_dir=CACHE_DIR):

        self.x_files = sorted(glob.glob(str(cache_dir / "*_MSI_simulated.npy")))
        self.y_files = sorted(glob.glob(str(cache_dir / "*_HSI_true.npy")))
        
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
    
class SpectralDatasetAug(Dataset):
    def __init__(self, cache_dir=CACHE_DIR, augment=False):
        self.augment = augment

        self.x_files = sorted(glob.glob(str(cache_dir / "*_MSI_simulated.npy")))
        self.y_files = sorted(glob.glob(str(cache_dir / "*_HSI_true.npy")))
        
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

    def augment_pair(self, x, y):
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[2])

        if random.random() < 0.5:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[1])

        k = torch.randint(0, 4, (1,)).item()
        x = torch.rot90(x, k, dims=[1,2])
        y = torch.rot90(y, k, dims=[1,2])

        return x, y

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, patch_idx = self.index_map[idx]
        
        x_patch = self.mmap_x_arrays[file_idx][patch_idx].copy()
        y_patch = self.mmap_y_arrays[file_idx][patch_idx].copy()

        x = torch.from_numpy(x_patch)
        y = torch.from_numpy(y_patch)

        if self.augment:
            x, y = self.augment_pair(x, y)

        return x, y
    
class IncertitudeDataset(Dataset):
    def __init__(self, cache_dir, augment=False):
        self.augment = augment

        self.msi_files = sorted(glob.glob(str(Path(cache_dir) / "*_MSI_simulated.npy")))
        self.hsi_true_files = sorted(glob.glob(str(Path(cache_dir) / "*_HSI_true.npy")))
        self.hsi_sim_files = sorted(glob.glob(str(Path(cache_dir) / "*_HSI_simulated.npy")))
        
        assert len(self.msi_files) == len(self.hsi_true_files) == len(self.hsi_sim_files), "Mismatch between MSI, HSI true and HSI simulated files."
        
        self.index_map = []
        self.mmap_msi = []
        self.mmap_hsi_true = []
        self.mmap_hsi_sim = []
        
        for f_idx, (f_msi, f_hsi_t, f_hsi_s) in enumerate(zip(self.msi_files, self.hsi_true_files, self.hsi_sim_files)):
            msi = np.load(f_msi, mmap_mode='r')
            hsi_t = np.load(f_hsi_t, mmap_mode='r')
            hsi_s = np.load(f_hsi_s, mmap_mode='r')
            
            self.mmap_msi.append(msi)
            self.mmap_hsi_true.append(hsi_t)
            self.mmap_hsi_sim.append(hsi_s)
            
            for p_idx in range(msi.shape[0]):
                self.index_map.append((f_idx, p_idx))

    def __len__(self):
        return len(self.index_map)

    def augment_triplet(self, x_msi, x_hsi, y_res):
        if random.random() < 0.5:
            x_msi = torch.flip(x_msi, dims=[2])
            x_hsi = torch.flip(x_hsi, dims=[2])
            y_res = torch.flip(y_res, dims=[2])

        if random.random() < 0.5:
            x_msi = torch.flip(x_msi, dims=[1])
            x_hsi = torch.flip(x_hsi, dims=[1])
            y_res = torch.flip(y_res, dims=[1])

        k = torch.randint(0, 4, (1,)).item()
        x_msi = torch.rot90(x_msi, k, dims=[1,2])
        x_hsi = torch.rot90(x_hsi, k, dims=[1,2])
        y_res = torch.rot90(y_res, k, dims=[1,2])

        return x_msi, x_hsi, y_res

    def __getitem__(self, idx):
        f_idx, p_idx = self.index_map[idx]
        
        msi = torch.from_numpy(self.mmap_msi[f_idx][p_idx].copy())
        hsi_true = torch.from_numpy(self.mmap_hsi_true[f_idx][p_idx].copy())
        hsi_sim = torch.from_numpy(self.mmap_hsi_sim[f_idx][p_idx].copy())

        X = torch.cat([msi, hsi_sim], dim=0)
        
        y = hsi_true - hsi_sim

        if self.augment:
            c_msi = msi.shape[0]
            msi_aug, hsi_sim_aug, y = self.augment_triplet(X[:c_msi], X[c_msi:], y)
            X = torch.cat([msi_aug, hsi_sim_aug], dim=0)

        return X , y    
    
def create_dataloaders(dataset_class, cache_dir=CACHE_DIR, batch_size=16, train_ratio=0.8, val_ratio=0.1, num_workers=4):
    
    train_base_ds = dataset_class(cache_dir, augment=True)
    eval_base_ds = dataset_class(cache_dir, augment=False)
    
    total_size = len(train_base_ds)
    indices = list(range(total_size))
    
    np.random.seed(42)  
    np.random.shuffle(indices)
    
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_dataset = Subset(train_base_ds, train_indices)
    val_dataset = Subset(eval_base_ds, val_indices)
    test_dataset = Subset(eval_base_ds, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"Splits -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader