# General
import os

# Imaging
import numpy as np
import torch

# Local
from src.model_utils import manage_model
from src.train_test_utils import train_test_utils


"""
Some found normalization parameters:

* Sentinel-1 IW1 calibrated sigma0:
    "eps_log": 0.007351449870059566,
    "s": 1.1081583499908447,

* TerraSAR-X Stripmap digital numbers:
    "eps_log": min(1., 735.6042847239643),
    "s": 1.2452837228775024,

"""


if __name__ == "__main__":

    # =========================================== #
    # ----- Definition of the training mode ----- #
    # =========================================== #
    args4loss = {"szcroploss": 0}

    train_mode = "merlin"

    if train_mode == "merlin":
        # --- No normalization:
        # norm_params = {
        #     "norm_fct_name": "def_norm_denorm_merlin_nonorm",
        #     "eps_log": 1.
        # }
        # --- Local normalization:
        norm_params = {
            "norm_fct_name": "def_norm_denorm_merlin_centloc",
            "eps_log": 0.007351449870059566,
            "s": 1.1081583499908447,
            "alpha": 1.,
            "mean_mode": "multiple_avgpool",  # can be "global", "avgpool" and "multiple_avgpool
            "kernel_size": 61,  # Used for "multiple_avgpool" and "avgpool"
            "n_avg_pool": 4,  # used for "multiple_avgpool"
        }

        mode_body = "unet"  # "unet" or "nafnet"
        args4body = {}

    elif train_mode == "var_diff_log_ref":
        # --- No normalization:
        # norm_params = {
        #     "norm_fct_name": "def_norm_denorm_merlin_nonorm",
        #     "eps_log": 1.
        # }
        # --- Local normalization:
        norm_params = {
            "norm_fct_name": "def_norm_denorm_sar_centloc",
            "eps_log": 1.,
            "s": 1.,
            "alpha": 1.,
            "mean_mode": "multiple_avgpool",  # can be "global", "avgpool" and "multiple_avgpool
            "kernel_size": 61,  # Used for "multiple_avgpool" and "avgpool"
            "n_avg_pool": 4,  # used for "multiple_avgpool"
        }

        mode_body = "nafnet"  # "unet" or "nafnet"
        args4body = {
            "nc_out_body": 32,  # Number of channels at the end of the body (before the head)
            "width": 16,
            "middle_blk_num": 1,
            "enc_blk_nums": [1, 1, 1, 8],
            "dec_blk_nums": [1, 1, 1, 1],
            "drop_out_rate": 0.
        }

    else:
        raise ValueError(f"Unrecognized 'train_mode': {train_mode}.")

    # ======================== #
    # ----- Saving paths ----- #
    # ======================== #
    curr_path = os.path.dirname(os.path.realpath(__file__))  # Current path
    save_folder = "sample"
    save_dir = os.path.join(curr_path, save_folder)


    # ====================== #
    # ----- Data paths ----- #
    # ====================== #
    data_path = curr_path
    # train_inp_data = "train/train_inp.npy"
    # train_tar_data = "train/train_tar.npy"
    # val_inp_data = "validation/val_inp.npy"
    # val_tar_data = "validation/val_tar.npy"
    train_inp_data = "train/numpy_array_RED_train_inputs.npy"
    train_tar_data = "train/numpy_array_RED_train_targets.npy"
    val_inp_data = "validation/numpy_array_RED_val_inputs.npy"
    val_tar_data = "validation/numpy_array_RED_val_targets.npy"


    # =============================== #
    # ----- Training parameters ----- #
    # =============================== #
    nb_epochs = 100
    nmax_iter_per_epoch = np.inf  # Set as few thousands if your dataset is too large
    nmax_iter_val_per_epoch = np.inf  # Should remain infinite
    patch_size = 256
    batch_size = 8

    lr_init = 0.0001
    lr_sched_mode = "cosine"  # "plateau", "cosine"
    lr_sched_args = None
    if lr_sched_mode == "plateau":
        lr_sched_args = {
            "factor": 0.5,
            "threshold": 1e-3,  # Must be adapted to the loss
            "patience": 10,  # Must be adapted to the number of epochs
            "min_lr": lr_init * 1e-3,
            "mode": "min",
            "threshold_mode": "abs"
        }
    elif lr_sched_mode == "cosine":
        lr_sched_args = {
            "T_0": 2,
            "T_mult": 2,
            "eta_min": 0.
        }

    loss2surv_lr = "val"  # "val", "train"   -   used for "plateau" learning-rate scheduler
    max_norm_grad = 5.0


    # ==================================== #
    # ----- Declaration of the model ----- #
    # ==================================== #
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = manage_model.get_model_from_archi_name(
        archi="basic",
        height=patch_size, width=patch_size,
        train_mode=train_mode, args4loss=args4loss, norm_params=norm_params,
        data_aug_intar=None, data_aug_in=None, data_aug_tar=None, device=device,
        mode_body=mode_body, args4body=args4body,
    )


    # =================================== #
    # ----- Launching the training ----- #
    # =================================== #
    history = train_test_utils.fit(
        model, data_path, save_dir, epochs=nb_epochs,
        train_inp_data=train_inp_data, train_tar_data=train_tar_data,
        val_inp_data=val_inp_data, val_tar_data=val_tar_data, mmap=True,
        batch_size=batch_size, eval_batch_size=1,
        nmax_iter_per_epoch=nmax_iter_per_epoch, nmax_iter_val_per_epoch=nmax_iter_val_per_epoch,
        save_all_best_train=False, save_all_best_val=False,
        lr_init=lr_init, lr_sched_mode=lr_sched_mode, lr_sched_args=lr_sched_args, loss2surv_lr=loss2surv_lr,
        max_norm_grad=max_norm_grad,
        isample_2keep=0
    )
