# General
import os
import warnings
from typing import Generator, Union

# Imaging
import numpy as np
import torch
import matplotlib.pyplot as plt

# Local
from src.manage_data import save_data
from src.processing import misc_process
from src.processing import normalization


# =================================================== #
# ----- Generator of intensities from a dataset ----- #
# =================================================== #

def generator_mode_realimag(
        files: list[str], mode_load: str, device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
) -> Generator[torch.Tensor, None, None]:

    if mode_load == "single_file":
        all_realimag = np.load(files[0], mmap_mode="r")
        shp = all_realimag.shape
        b, c = misc_process.get_bc_to_manage_shape(shp, nc=2)
        if b is None:
            realimag = torch.tensor(all_realimag, device=device)
            realimag =  misc_process.manage_shape_torch_from_bc(realimag, b, c)
            intensity = torch.square(realimag[:, 0:1, ...]) + torch.square(realimag[:, 1:2, ...])
            yield intensity

        else:
            for i in range(shp[b]):
                slc = [slice(None) if j != b else slice(i, i+1) for j in range(len(shp))]
                realimag = torch.tensor(all_realimag[*slc], device=device)
                realimag =  misc_process.manage_shape_torch_from_bc(realimag, b, c)
                intensity = torch.square(realimag[:, 0:1, ...]) + torch.square(realimag[:, 1:2, ...])
                yield intensity

    else:
        for i in range(len(files)):
            realimag = torch.tensor(np.load(files[i]), device=device)
            realimag = misc_process.manage_shape_torch(realimag, nc=2)
            intensity = torch.square(realimag[:, 0:1, ...]) + torch.square(realimag[:, 1:2, ...])
            yield intensity


def generator_mode_intensity(
        files: list[str], mode_load: str, device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
) -> Generator[torch.Tensor, None, None]:

    if mode_load == "single_file":
        all_intensities = np.load(files[0], mmap_mode="r")
        shp = all_intensities.shape
        b, c = misc_process.get_bc_to_manage_shape(shp, nc=1)
        if b is None:
            intensity = torch.tensor(all_intensities, device=device)
            intensity =  misc_process.manage_shape_torch_from_bc(intensity, b, c)
            yield intensity

        else:
            for i in range(shp[b]):
                slc = [slice(None) if j != b else slice(i, i+1) for j in range(len(shp))]
                intensity = torch.tensor(all_intensities[*slc], device=device)
                intensity =  misc_process.manage_shape_torch_from_bc(intensity, b, c)
                yield intensity

    else:
        for i in range(len(files)):
            intensity = torch.tensor(np.load(files[i]), device=device)
            intensity = misc_process.manage_shape_torch(intensity, nc=1)
            yield intensity


def generator_mode_amplitude(
        files: list[str], mode_load: str, device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
) -> Generator[torch.Tensor, None, None]:

    if mode_load == "single_file":
        all_amplitudes = np.load(files[0], mmap_mode="r")
        shp = all_amplitudes.shape
        b, c = misc_process.get_bc_to_manage_shape(shp, nc=1)
        if b is None:
            amplitude = torch.tensor(all_amplitudes, device=device)
            amplitude =  misc_process.manage_shape_torch_from_bc(amplitude, b, c)
            intensity = torch.square(amplitude)
            yield intensity

        else:
            for i in range(shp[b]):
                slc = [slice(None) if j != b else slice(i, i+1) for j in range(len(shp))]
                amplitude = torch.tensor(all_amplitudes[*slc], device=device)
                amplitude =  misc_process.manage_shape_torch_from_bc(amplitude, b, c)
                intensity = torch.square(amplitude)
                yield intensity

    else:
        for i in range(len(files)):
            amplitude = torch.tensor(np.load(files[i]), device=device)
            intensity = torch.square(amplitude)
            intensity = misc_process.manage_shape_torch(intensity, nc=1)
            yield intensity


def generator_mode_real_n_imag(
        files: tuple[list[str], list[str]], mode_load: str, device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
) -> Generator[torch.Tensor, None, None]:

    if mode_load == "single_file":
        all_real = np.load(files[0][0], mmap_mode="r")
        all_imag = np.load(files[1][0], mmap_mode="r")
        shp = all_real.shape
        b, c = misc_process.get_bc_to_manage_shape(shp, nc=1)
        if b is None:
            real = torch.tensor(all_real, device=device)
            imag = torch.tensor(all_imag, device=device)
            real =  misc_process.manage_shape_torch_from_bc(real, b, c)
            imag =  misc_process.manage_shape_torch_from_bc(imag, b, c)
            intensity = torch.square(real) + torch.square(imag)
            yield intensity

        else:
            for i in range(shp[b]):
                slc = [slice(None) if j != b else slice(i, i+1) for j in range(len(shp))]
                real = torch.tensor(all_real[*slc], device=device)
                imag = torch.tensor(all_imag[*slc], device=device)
                real =  misc_process.manage_shape_torch_from_bc(real, b, c)
                imag =  misc_process.manage_shape_torch_from_bc(imag, b, c)
                intensity = torch.square(real) + torch.square(imag)
                yield intensity

    else:
        for i in range(len(files[0])):
            real = torch.tensor(np.load(files[0][i]), device=device)
            imag = torch.tensor(np.load(files[1][i]), device=device)
            intensity = torch.square(real) + torch.square(imag)
            intensity = misc_process.manage_shape_torch(intensity, nc=1)
            yield intensity


def manage_mode(
        mode: str, files: Union[str, list[str], list[list[str]]], data_path: str = ""
) -> tuple[str, Union[list[str], tuple[list[str], list[str]]]]:

    if mode == "real_n_imag":
        assert (isinstance(files, Union[list, tuple])) and (len(files) == 2), (
            "You must provide a list of two files (or a list of two lists of files) for the mode 'real_n_imag'.")

        if isinstance(files[0], str):
            if os.path.isfile(os.path.join(data_path, files[0])):
                mode_load = "single_file"
                list_files = [[files[0]], [files[1]]]
            else:
                all_files_real = [os.path.join(files[0], f) for f in os.listdir(os.path.join(data_path, files[0]))]
                all_files_imag = [os.path.join(files[1], f) for f in os.listdir(os.path.join(data_path, files[1]))]
                all_files_real.sort()
                all_files_imag.sort()
                mode_load = "multi_file" if (len(all_files_real) > 1) else "single_file"
                list_files = [all_files_real, all_files_imag]

        else:
            list_files = files
            mode_load = "multi_file" if (len(list_files[0]) > 1) else "single_file"

    else:

        if isinstance(files, str):
            if os.path.isfile(os.path.join(data_path, files)):
                mode_load = "single_file"
                list_files = [files]
            else:
                list_files = [os.path.join(files, f) for f in os.listdir(os.path.join(data_path, files))]
                mode_load = "multi_file" if (len(list_files) > 1) else "single_file"

        else:
            list_files = files
            mode_load = "multi_file" if (len(list_files) > 1) else "single_file"

    return mode_load, list_files


def get_generator_of_intensities(
        mode: str, files: Union[str, list[str], list[list[str]]], data_path: str = "",
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
) -> Generator[torch.Tensor, None, None]:

    mode_load, list_files = manage_mode(mode, files, data_path)

    if mode == "realimag":
        return generator_mode_real_n_imag(list_files, mode_load, device)
    elif mode == "intensity":
        return generator_mode_intensity(list_files, mode_load, device)
    elif mode == "amplitude":
        return generator_mode_amplitude(list_files, mode_load, device)
    elif mode == "real_n_imag":
        return generator_mode_real_n_imag(list_files, mode_load, device)
    else:
        raise ValueError(f"Unknown mode '{mode}'.")


# ================================================================================================================= #
# ----- Searching for normalization constants: thermal noise level and useful dynamic of log-intensity images ----- #
# ================================================================================================================= #

def gen_visu_thnoise(avg_intensity: torch.Tensor, thermal_noise: float) -> np.ndarray:
    ampl = torch.sqrt(avg_intensity)
    m3std = torch.mean(ampl) + 3 * torch.std(ampl)
    np_ampl = ampl.permute(0, 2, 3, 1).cpu().numpy()
    rbg_ampl = np.clip(np_ampl / m3std.item(), 0, 1).repeat(3, -1)
    rbg_ampl[np_ampl[..., 0]**2 <= 5*thermal_noise, 0] = 0.4
    rbg_ampl[np_ampl[..., 0]**2 <= 2*thermal_noise, 0] = 0.75
    rbg_ampl[np_ampl[..., 0]**2 <= thermal_noise, 0] = 1
    rbg_ampl = (np.clip(rbg_ampl, a_min=0, a_max=1) * 255).astype(np.uint8)
    return rbg_ampl


def plot_visu_thnoise(avg_intensity: torch.Tensor, thermal_noise: float, nb_pix_per_lev: int = 10000) -> plt.Figure:
    quantiz_intens = np.prod(np.array(avg_intensity.shape)) // nb_pix_per_lev
    frq, edges = np.histogram(
        avg_intensity.cpu().numpy().flatten(), bins=quantiz_intens, range=(0, torch.mean(avg_intensity).item()))

    rbg_ampl = gen_visu_thnoise(avg_intensity, thermal_noise)

    fig = plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1), plt.imshow(rbg_ampl[0, ...], vmin=0, vmax=255)
    plt.subplot(1, 2, 2), plt.bar(edges[:-1], frq, width=np.diff(edges))
    plt.vlines([thermal_noise], ymin=0, ymax=np.max(frq), linestyle="dashed", colors="red")
    plt.xlim(0, edges[-2])
    plt.tight_layout()

    return fig


def search_thermal_noise_level(
        intensity: torch.Tensor, window_size: int = 11, nb_pix_per_lev: int = 10000, pct_ech_min: float = None,
        nb_lev_min: int = 5, raise_exeption: bool = True, visualize_thnoise: bool = False
) -> float:

    avg_intensity = normalization.avg_pool_2d_with_avg_pad(intensity, kernel_size=window_size)
    nbp = int(np.prod(np.array(avg_intensity.shape)))

    # First bin to decrease is considered as the thermal noise level.
    # The intuition is that all black areas (water, shadows, etc.) will have the same distribution of the thermal
    # noise only, and that all other areas will have different distribution which minimum is higher.
    quantiz_intens = np.prod(np.array(avg_intensity.shape)) // nb_pix_per_lev
    frq, edges = np.histogram(
        avg_intensity.cpu().numpy().flatten(), bins=quantiz_intens, range=(0, torch.mean(avg_intensity).item()))
    thermal_noise = None
    i = nb_lev_min
    while i < len(frq) and (thermal_noise is None):
        if (frq[i] < frq[i-1]) and (frq[i-1] > pct_ech_min * nbp):
            thermal_noise = (edges[i] + edges[i-1]) / 2
        i += 1

    if raise_exeption:
        assert thermal_noise is not None, "Discretization is not fine enought."
    elif thermal_noise is None:
        warnings.warn("Discretization is not fine enought.")

    if visualize_thnoise and (thermal_noise is not None):
        plot_visu_thnoise(avg_intensity, thermal_noise, nb_pix_per_lev)
        plt.show()

    return thermal_noise


def search_dynamic(intensity: torch.Tensor, thnoise: float) -> float:
    return torch.std(torch.log(intensity + thnoise)).cpu().numpy().item()


def check_succesiv_mean(
        l: np.ndarray, nb_min: int = 3, fact_thresh: float = 2., do_median: bool = False
) -> tuple[float, int]:

    mean_or_med = np.median if do_median else np.mean
    l = np.sort(l)

    best_med = float(mean_or_med(l[:nb_min]))
    i = nb_min
    too_far = False
    while (i < len(l)) and (not too_far):
        if l[i] > fact_thresh * best_med:
            too_far = True
        else:
            i += 1
            best_med = float(mean_or_med(l[:i]))

    return best_med, i-1


def search_useful_dynamic_with_files(
        mode: str, files: Union[str, list[str], list[list[str]]], data_path: str = "",
        window_size_4thn: int = 11, nb_pix_per_lev_4thn: int = 10000, nb_lev_min_4thn: int = 5,
        pct_ech_min_4thn: float = 1e-4, nb_min_thns: int = 3, fact_thresh_thns: float = 2.,
        raise_exeption_4thn: bool = True, visualize_thnoise: bool = False, visualize_stats: bool = False,
        path4save: str = "none", suff4save: str = "",
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
) -> tuple[float, float]:

    print(f"\nEstimation of the thermal noise level...")
    generator_intensities = get_generator_of_intensities(mode=mode, files=files, data_path=data_path, device=device)
    thns = []
    for intens in generator_intensities:
        thn = search_thermal_noise_level(
            intens,
            window_size=window_size_4thn, nb_pix_per_lev=nb_pix_per_lev_4thn, pct_ech_min=pct_ech_min_4thn,
            nb_lev_min=nb_lev_min_4thn, raise_exeption=raise_exeption_4thn, visualize_thnoise=False)
        if thn is not None:
            thns.append(thn)

    thns = np.array([thn for thn in thns if thn is not None])
    best_thn_succmean, isuccmean = check_succesiv_mean(thns, fact_thresh=fact_thresh_thns, nb_min=nb_min_thns)

    print(f"\nEstimation of the parameter s (standard deviation of log-intensities)...")
    generator_intensities = get_generator_of_intensities(mode=mode, files=files, data_path=data_path, device=device)
    s_s = []
    for intens in generator_intensities:
        s_s.append(search_dynamic(intens, best_thn_succmean))
    s = float(np.mean(s_s))

    best_thn_succmed, isuccmed = check_succesiv_mean(
        thns, fact_thresh=fact_thresh_thns, nb_min=nb_min_thns, do_median=True)
    print(f"\nEstimation of the thermal noise:\n"
          f"     Median             = {np.median(thns):.6f}\n"
          f"     Average            = {np.mean(thns):.6f}\n"
          f"     Standard-deviation = {np.std(thns):.6f}\n"
          f"     Successive mean    = {best_thn_succmean:.6f} ({isuccmean + 1} lowest > {np.sort(thns)[:isuccmean + 1]}) (chosen method)\n"
          f"     Successive median  = {best_thn_succmed:.6f} ({isuccmed + 1} lowest > {np.sort(thns)[:isuccmed + 1]})\n"
          )
    print(f"\nEstimation of the parameter s:\n"
          f"     Median             = {np.median(s_s):.6f}\n"
          f"     Average            = {np.mean(s_s):.6f} (chosen method)\n"
          f"     Standard-deviation = {np.std(s_s):.6f}\n"
          )

    if visualize_stats:
        plt.figure(figsize=(16, 8), dpi=200)
        sthns = np.sort(thns)
        plt.plot(np.sort(thns))
        plt.plot([np.mean(sthns[:i]) for i in range(1, len(sthns)+1)])
        plt.plot([np.median(sthns[:i]) for i in range(1, len(sthns)+1)])
        plt.plot([np.nan] + [sthns[i] - float(np.mean(sthns[:i])) for i in range(1, len(sthns))])
        plt.plot([np.nan] + [sthns[i] - float(np.median(sthns[:i])) for i in range(1, len(sthns))])
        plt.vlines([isuccmean], ymin=0, ymax=np.max(thns), colors="red", linestyles="dashed")
        plt.ylim(0, np.max(thns))
        plt.legend(["Sorted thermal noise levels", "Successive mean", "Successive median",
                    "current - fact * previous successive mean", "current - fact * previous successive median"])
        plt.title("Repartition function of thermal noise estimations")
        plt.tight_layout()
        if (path4save is not None) and (path4save not in ["n", "none"]):
            plt.savefig(os.path.join(path4save, f"norm_params_cumulative_esti_thn{suff4save}.png"))

        plt.figure(figsize=(16, 8), dpi=200)
        sthns = np.sort(thns)
        plt.plot([np.nan] + [sthns[i] / fact_thresh_thns / float(np.mean(sthns[:i])) for i in range(1, len(sthns))])
        plt.plot([np.nan] + [sthns[i] / fact_thresh_thns / float(np.median(sthns[:i])) for i in range(1, len(sthns))])
        plt.hlines([1.], xmin=0, xmax=len(sthns), colors="black", linestyles="dashed")
        plt.vlines([isuccmean], ymin=0.5, ymax=2., colors="red", linestyles="dashed")
        plt.ylim(0.5, 2.)
        plt.legend(["current / (fact * previous successive mean)", "current / (fact * previous successive median)"])
        plt.title("Ratio to be above 1 to stop the search of the thermal noise level")
        plt.tight_layout()
        if (path4save is not None) and (path4save != "none"):
            plt.savefig(os.path.join(path4save, f"ratio_to_threshold_thn{suff4save}.png"))

        plt.figure(figsize=(16, 8), dpi=200)
        plt.plot(np.sort(s_s))
        plt.ylim(0, 1.1*np.max(s_s))
        plt.title("Increasing repartition of standard deviation of log-intensity")
        plt.tight_layout()
        if (path4save is not None) and (path4save != "none"):
            plt.savefig(os.path.join(path4save, f"norm_params_repart_func_std_log_int{suff4save}.png"))

    if visualize_thnoise:
        generator_intensities = get_generator_of_intensities(mode=mode, files=files, data_path=data_path, device=device)
        i = 0
        for intens in generator_intensities:
            avg_intens = normalization.avg_pool_2d_with_avg_pad(intens, kernel_size=window_size_4thn)
            fig = plot_visu_thnoise(avg_intens, best_thn_succmean, nb_pix_per_lev_4thn)
            if (path4save is not None) and (path4save != "none"):
                fig.savefig(os.path.join(path4save, f"illust_thn_image_{i}"))
            else:
                fig.savefig(os.path.join(data_path, f"illust_thn_image_{i}"))
            i += 1

    return best_thn_succmean, s
