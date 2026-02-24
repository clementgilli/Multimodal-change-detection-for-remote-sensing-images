# General
import os
from typing import Union
import warnings
import csv
import datetime
from tqdm import tqdm

# Imaging
import numpy as np
import torch
import matplotlib.pyplot as plt

# Local
from src.manage_data import dataset
from src.misc import count_cpu
from src.misc import misc_xml
from src.model_utils import general_model
from src.model_utils import manage_model


# def save_model_path(model: torch.nn.Module, path: str):
    # torch.save(model, path)
def save_model_path(model: Union[torch.nn.Module, general_model.Model], path: str):
    if isinstance(model, general_model.Model):
        model.save_hole_model(path)
    else:
        torch.save(model, path)
        # torch.save(model.state_dict(), path)


def save_model(model, destination_folder, ep_num=-1):
    """
      save the ".pth" model_utils in destination_folder
    """

    if ep_num == -1:
        path = os.path.join(destination_folder, "trained_model.pth")
        save_model_path(model, path)
        path = os.path.join(destination_folder, "trained_model_best_val.pth")
    elif ep_num == -2:
        path = os.path.join(destination_folder, "trained_model_best_train.pth")
    elif ep_num == -3:
        path = os.path.join(destination_folder, "trained_model_end.pth")
    else:
        path = os.path.join(destination_folder, "trained_model_ep"+str(ep_num)+".pth")

    save_model_path(model, path)


def load_model_path(path: str, model: torch.nn.Module = None):
    # return torch.load(path, weights_only=False)

    if os.path.isfile(path.replace(".pth", ".xml")):
        model_params = misc_xml.load_xml_to_dict(path.replace(".pth", ".xml"))
    else:
        model_params = None

    if (model is None) and ((model_params is None) or (model_params.get("class", None) is None)):
        raise AssertionError(f"Impossible to load a model if 'model' is None and if "
                             f"'{path.replace('.pth', '.xml').split('/')[-1]}' is not available or if it "
                             f"doesn't provide the class of the model.")

    elif (model is not None) or (model_params is None) or (model_params.get("class", None) is None):
        pass

    else:
        model_args = dict(model_params["gen_mod_params"], **model_params["archi_params"])
        class_str = model_params["class"]
        class_str = class_str.split(".")[-2]
        model = manage_model.get_model_from_archi_name(class_str, **model_args)

    model.load_state_dict(torch.load(path, map_location=model.device))

    return model


def save_history_dict(history_dict: dict, destination_folder: str, csv_name: str = "history"):
    # header
    header = list(history_dict.keys())
    # rows
    rows = zip(*[v1 for k1, v1 in history_dict.items()])
    # write
    with open(os.path.join(destination_folder, csv_name+".csv"), "w") as ofile:
        wr = csv.writer(ofile, dialect="excel")
        wr.writerow(header)
        wr.writerows(rows)


def gen_histo_csv_from_cmd(
        path2sample: str, cmd_file: str,
        sepb: str = ": ", sepe: str = " - ", flg_train: str = "training loss", flg_val: str = "validation loss"
):
    f = open(os.path.join(path2sample, cmd_file), "r")
    history = {
        "train_loss": [],
        "val_loss": [],
    }
    for l in f.readlines():
        l = l.strip("\n")
        if flg_train in l.lower():
            history["train_loss"].append(float(l.split(sepb)[-1].split(sepe)[0]))
        elif flg_val in l.lower():
            history["val_loss"].append(float(l.split(sepb)[-1].split(sepe)[0]))

    history["train_loss"] = history["train_loss"][:len(history["val_loss"])]
    save_history_dict(history, path2sample, csv_name="history_from_outcmd")


def load_history(path2sample: str, auth_from_cmd: bool = True):
    pathcsv = os.path.join(path2sample, "history.csv")
    if (not os.path.exists(pathcsv)) and auth_from_cmd:
        if os.path.isfile(os.path.join(path2sample, "history_from_outcmd.csv")):
            pathcsv = os.path.join(path2sample, "history_from_outcmd.csv")
        else:
            cmd_file = [f for f in os.listdir(path2sample) if (f.endswith(".out") or (f == "cmd.txt"))][0]
            assert os.path.exists(os.path.join(path2sample, cmd_file)), f"No history or output file in {path2sample}."
            gen_histo_csv_from_cmd(path2sample, cmd_file)
            pathcsv = os.path.join(path2sample, "history_from_outcmd.csv")

    with open(pathcsv, "r") as ofile:
        lines = [ll[:-1] for ll in ofile.readlines()]
        keys, values = lines[0].split(","), lines[1:]

        values = np.array([[float(vstr) for vstr in ll.split(",")] for ll in values])

        histo = dict()
        for ik in range(len(keys)):
            histo[keys[ik]] = values[:, ik]
        return histo


def save_training_infos(save_dir: str, **kwargs):
    xml_str = misc_xml.dict_to_xml("training_infos", kwargs)
    misc_xml.save_xml_str(os.path.join(save_dir, "training_infos.xml"), xml_str)


def fit(
        model: torch.nn.Module, data_path: str, save_dir: str, epochs: int,
        train_inp_data: str = "train_inp", train_tar_data: str = "train_tar",
        val_inp_data: str = "validation_inp", val_tar_data: str = "validation_tar", mmap: bool = True,
        batch_size: int = 16, eval_batch_size: int = 1,
        nmax_iter_per_epoch: int = np.inf, nmax_iter_val_per_epoch: int = np.inf,
        save_all_best_train: bool = False, save_all_best_val: bool = False,
        lr_init: float = 0.0001, lr_sched_mode: str = "plateau", lr_sched_args: dict = None, loss2surv_lr: str = "val",
        max_norm_grad: float = 5.0,
        isample_2keep: int = 0
) -> dict:

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_training_infos(
        save_dir=save_dir, data_path=data_path, epochs=epochs,
        train_inp_data=train_inp_data, train_tar_data=train_tar_data,
        val_inp_data=val_inp_data, val_tar_data=val_tar_data, mmap=mmap,
        batch_size=batch_size, eval_batch_size=eval_batch_size,
        nmax_iter_per_epoch=nmax_iter_per_epoch, nmax_iter_val_per_epoch=nmax_iter_val_per_epoch,
        save_all_best_train=save_all_best_train, save_all_best_val=save_all_best_val,
        lr_init=lr_init, lr_sched_mode=lr_sched_mode, lr_sched_args=lr_sched_args, loss2surv_lr=loss2surv_lr,
        max_norm_grad=max_norm_grad, isample_2keep=isample_2keep
    )

    # ----- Preparation of the dataset ----- #
    if os.path.isdir(os.path.join(data_path, train_inp_data)):
        train_data = dataset.Dataset(
            os.path.join(data_path, train_inp_data), os.path.join(data_path, train_tar_data), mmap=mmap)
    else:
        train_data = dataset.PackedDataset(
            os.path.join(data_path, train_inp_data), os.path.join(data_path, train_tar_data), mmap=mmap)

    if os.path.isdir(os.path.join(data_path, val_inp_data)):
        val_data = dataset.Dataset(
            os.path.join(data_path, val_inp_data), os.path.join(data_path, val_tar_data), mmap=mmap)
    else:
        val_data = dataset.PackedDataset(
            os.path.join(data_path, val_inp_data), os.path.join(data_path, val_tar_data), mmap=mmap)

    # ----- Preparation of the dataloader ----- #
    try:
        nworkers = count_cpu.available_cpu_count()
    except Exception as e:
        warnings.warn(f"Error inside 'count_cpu.available_cpu_count()':\n{e}\n num_workers defined as 4.")
        nworkers = 8
    nworkers = min(nworkers-1, 8)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=nworkers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=eval_batch_size, shuffle=False, drop_last=True, num_workers=nworkers, pin_memory=True)

    train_loader_iter = iter(train_loader)
    val_loader_iter = iter(val_loader)

    # ----- Optimizer and learning-rate scheduler ----- #
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    if lr_sched_args is None:
        lr_sched_args = dict()

    if lr_sched_mode == "plateau":
        lr_sched_args_base = {"factor": 0.2, "threshold": 1., "patience": max(4, epochs // 10),
                              "min_lr": lr_init * 0.01, "mode": "min", "threshold_mode": "abs"}
        for k in lr_sched_args_base.keys():
            lr_sched_args[k] = lr_sched_args.get(k, lr_sched_args_base.get(k))

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **lr_sched_args)

    elif lr_sched_mode == "cosine":
        lr_sched_args_base = {"T_0": max(1, epochs // 50), "T_mult": 2, "eta_min": 0.}
        for k in lr_sched_args_base.keys():
            lr_sched_args[k] = lr_sched_args.get(k, lr_sched_args_base.get(k))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **lr_sched_args)

    else:
        scheduler = torch.optim.lr_scheduler.LRScheduler(optimizer)

    # ----- Training loop ----- #
    train_losses = np.zeros([epochs], dtype=np.float32)
    nb_batch_train = int(np.ceil(len(train_data) / batch_size))
    nit_per_ep = min(nb_batch_train, nmax_iter_per_epoch)
    min_loss_train = np.inf

    val_losses = np.zeros([epochs], dtype=np.float32)
    nb_batch_val = int(np.ceil(len(val_data) / eval_batch_size))
    nit_val_per_ep = min(nb_batch_val, nmax_iter_val_per_epoch)
    min_loss_val = np.inf
    idx_batch_val_2keep = min(isample_2keep, nit_val_per_ep - 1)

    lrs = np.zeros([epochs])

    epoch_num = 0
    for epoch in range(epochs):
        beg_epoch_time = datetime.datetime.now()

        epoch_num += 1
        print("\nEpoch", epoch + 1)

        lrs[epoch] = scheduler.get_last_lr()[0]
        print(f"Learning-rate: {lrs[epoch]}")

        # Train
        train_losses_curr = torch.zeros(nit_per_ep, dtype=torch.float32, device=model.device)
        num_batch = 0
        for _ in tqdm(range(nit_per_ep)):
            try:
                batch = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)
                batch = next(train_loader_iter)

            optimizer.zero_grad()
            loss = model.training_step(batch, num_batch)
            train_losses_curr[num_batch] = loss

            loss.backward()

            if (max_norm_grad is not None) and (max_norm_grad != np.inf):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm_grad)

            optimizer.step()

            num_batch += 1

        train_loss = torch.mean(train_losses_curr).detach().cpu().numpy()
        train_losses[epoch] = train_loss

        if train_loss < min_loss_train:
            min_loss_train = train_loss
            if save_dir is not None:
                if save_all_best_train:
                    save_model(model, save_dir, ep_num=epoch + 1)
                else:
                    save_model(model, save_dir, ep_num=-2)
                print("New best training loss:", min_loss_train, "- Model saved at epoch", epoch + 1)
            else:
                print("New best training loss:", min_loss_train)
        else:
            print("Training Loss:", train_loss)

        # Validate
        with torch.no_grad():
            val_losses_curr = torch.zeros(nit_val_per_ep, dtype=torch.float32, device=model.device)
            num_batch = 0
            for _ in tqdm(range(nit_val_per_ep)):
                try:
                    batch = next(val_loader_iter)
                except StopIteration:
                    val_loader_iter = iter(val_loader)
                    batch = next(val_loader_iter)

                vloss = model.validation_step(
                    batch, num_batch, epoch_num, save_dir, sample_batch_number=idx_batch_val_2keep)
                val_losses_curr[num_batch] = vloss

                num_batch += 1

        val_loss = torch.mean(val_losses_curr).cpu().numpy()
        val_losses[epoch] = val_loss

        if val_loss < min_loss_val:
            min_loss_val = val_loss
            if save_dir is not None:
                if save_all_best_val:
                    save_model(model, save_dir, ep_num=epoch + 1)
                else:
                    save_model(model, save_dir, ep_num=-1)
                print("New best validation loss:", min_loss_val, "- Model saved at epoch", epoch + 1)
            else:
                print("New best validation loss:", min_loss_val)
        else:
            print("Validation Loss:", val_loss)

        if lr_sched_mode == "plateau":
            loss2surv = train_loss if ("train" in loss2surv_lr) else val_loss
            scheduler.step(loss2surv)  # The scheduler is able to decrease the learning rate
        else:
            scheduler.step()

        print("Duration Epoch:", datetime.datetime.now() - beg_epoch_time)

    save_model(model, save_dir, ep_num=-3)  # Saving last epoch weights
    print("\n Trained model saved at", save_dir, "as trained_model.pth")

    # ----- Saving the history ----- #
    history = {"train_loss": list(train_losses), "val_loss": list(val_losses), "lrs": list(lrs)}
    save_history_dict(history, save_dir)
    print("History saved!")

    frang = 0.05
    all_losses = np.stack([train_losses, val_losses], axis=0)
    mini = np.min(all_losses)
    maxi = np.quantile(all_losses, 0.95)
    rang = maxi-mini
    mini, maxi = mini - frang*rang, maxi + frang*rang

    if not np.isnan(mini):
        plt.figure()
        plt.plot(range(1, epochs+1), train_losses)
        plt.plot(range(1, epochs+1), val_losses)
        plt.xlim(1, epochs)
        plt.ylim(mini, maxi)
        plt.title("Train and Validation Losses")
        plt.legend(["Train Loss", "Validation Loss"])
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_loss"), dpi=300)

        if mini > 0:
            plt.yscale("log")
        else:
            plt.figure()
            plt.yscale("log")
            plt.plot(range(1, epochs+1), train_losses - mini)
            plt.plot(range(1, epochs+1), val_losses - mini)
            plt.xlim(1, epochs)
            plt.ylim(-1, maxi-mini)  # -1 automatically adapts
        plt.title("Normalized Train and Validation Losses in log-scale")
        plt.legend(["Train Loss", "Validation Loss"])
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_loss_log"), dpi=300)

        plt.figure()
        plt.plot(range(1, epochs+1), lrs)
        plt.title("Learning-rate")
        plt.savefig(os.path.join(save_dir, "training_learning_rate"), dpi=300)

        plt.draw()

        plt.close("all")

    return history
