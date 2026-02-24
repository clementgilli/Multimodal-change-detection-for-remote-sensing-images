# General
import warnings
from typing import Any, Union

# Imaging
import numpy as np
import torch
import cv2

# Local


# ********************************************************************** #
#                                                                        #
# ---------- Helpful normalization functions on torch tensors ---------- #
#                                                                        #
# ********************************************************************** #

# ================================ #
# ----- Basic normalizations ----- #
# ================================ #

# --- Normalization
def norm_tensor(t: torch.Tensor, m: float = 0., M: float = 1.) -> torch.Tensor:
    return (t - m) / (M - m)

def denorm_tensor(t: torch.Tensor, m: float = 0., M: float = 1.) -> torch.Tensor:
    return t * (M - m) + m


# --- Offset
def ofst_tensor(t: torch.Tensor, m: float = 0.) -> torch.Tensor:
    return t - m

def deofst_tensor(t: torch.Tensor, m: float = 0.) -> torch.Tensor:
    return t + m


# --- Dilatation
def dilat_tensor(t: torch.Tensor, M: float = 1.) -> torch.Tensor:
    return t / M

def dedilat_tensor(t: torch.Tensor,  M: float = 1.) -> torch.Tensor:
    return t * M


# --- Computation of log-intensity without normalization
def ampl_to_log_intens_t(t: torch.Tensor, eps_log: float = 1e-3) -> torch.Tensor:
    return torch.log(t**2 + eps_log)

def ampl_to_log_intens_clip_eps_t(t: torch.Tensor, eps_log: float = 1e-3) -> torch.Tensor:
    return torch.log(torch.clamp(t**2, min=eps_log))

def log_intens_to_intens_t(t: torch.Tensor) -> torch.Tensor:
    return torch.exp(t)

def log_intens_to_ampl_t(t: torch.Tensor) -> torch.Tensor:
    return torch.exp(0.5 * t)


# --- Normalization of log-intensity
def ampl_to_norm_log_intens_t(
        t: torch.Tensor, eps_log: float = 1e-3, m: float = -1.429329123112601, M: float = 10.089038980848645
) -> torch.Tensor:
    return (torch.log(t**2 + eps_log) - 2 * m) / (2 * (M - m))

def ampl_to_norm_log_intens_clip_eps_t(
        t: torch.Tensor, eps_log: float = 1e-3, m: float = -1.429329123112601, M: float = 10.089038980848645
) -> torch.Tensor:
    return (torch.log(torch.clamp(t**2, min=eps_log)) - 2 * m) / (2 * (M - m))

def norm_log_intens_t(
        t: torch.Tensor, m: float = -1.429329123112601, M: float = 10.089038980848645
) -> torch.Tensor:
    return (t - 2 * m) / (2 * (M - m))

def denorm_log_intens_t(
        t: torch.Tensor, m: float = -1.429329123112601, M: float = 10.089038980848645
) -> torch.Tensor:
    return t * 2 * (M - m) + 2 * m

def norm_log_intens_to_intens_t(
        t: torch.Tensor, m: float = -1.429329123112601, M: float = 10.089038980848645
) -> torch.Tensor:
    return torch.exp(t * 2 * (M - m) + 2 * m)

def norm_log_intens_to_ampl_t(
        t: torch.Tensor, m: float = -1.429329123112601, M: float = 10.089038980848645
) -> torch.Tensor:
    return torch.exp(t * (M - m) + m)


# --- Offset of log-intensity only
def ampl_to_ofst_log_intens_t(t: torch.Tensor, eps_log: float = 1e-3, m: float = -1.429329123112601) -> torch.Tensor:
    return torch.log(t**2 + eps_log) - 2 * m

def ampl_to_ofst_log_intens_clip_eps_t(t: torch.Tensor, eps_log: float = 1e-3, m: float = -1.429329123112601) -> torch.Tensor:
    return torch.log(torch.clamp(t**2, min=eps_log)) - 2 * m

def ofst_log_intens_t(t: torch.Tensor, m: float = -1.429329123112601) -> torch.Tensor:
    return t - 2 * m

def deofst_log_intens_t(t: torch.Tensor, m: float = -1.429329123112601) -> torch.Tensor:
    return t + 2 * m

def ofst_log_intens_to_intens_t(t: torch.Tensor, m: float = -1.429329123112601) -> torch.Tensor:
    return torch.exp(t + 2 * m)

def ofst_log_intens_to_ampl_t(t: torch.Tensor, m: float = -1.429329123112601) -> torch.Tensor:
    return torch.exp(0.5 * t + m)


# --- Dilatation of log-intensity only
def ampl_to_dilat_log_intens_t(t: torch.Tensor, eps_log: float = 1e-3, M: float = 10.089038980848645) -> torch.Tensor:
    return torch.log(t**2 + eps_log) / (2 * M)

def ampl_to_dilat_log_intens_clip_eps_t(t: torch.Tensor, eps_log: float = 1e-3, M: float = 10.089038980848645) -> torch.Tensor:
    return torch.log(torch.clamp(t**2, min=eps_log)) / (2 * M)

def dilat_log_intens_t(t: torch.Tensor, M: float = 10.089038980848645) -> torch.Tensor:
    return t / 2 * M

def dedilat_log_intens_t(t: torch.Tensor, M: float = 10.089038980848645) -> torch.Tensor:
    return t * 2 * M

def dilat_log_intens_to_intens_t(t: torch.Tensor, M: float = 10.089038980848645) -> torch.Tensor:
    return torch.exp(t * 2 * M)

def dilat_log_intens_to_ampl_t(t: torch.Tensor, M: float = 10.089038980848645) -> torch.Tensor:
    return torch.exp(t * M)


# ================================================= #
# ----- Local and more complex normalizations ----- #
# ================================================= #

def avg_pool_2d_with_avg_pad(t: torch.Tensor, kernel_size: int) -> torch.Tensor:

    # kernel_size nust be odd so the output has the same size as the input
    assert kernel_size % 2 == 1, "Please provide an odd kernel size."

    t = avg_pad_2d(t, pad=kernel_size//2)
    return torch.nn.functional.avg_pool2d(t, kernel_size=kernel_size, padding=0, stride=1)


def center_each_batch_chan(t: torch.Tensor) -> torch.Tensor:
    return t - torch.mean(t, dim=(-2, -1), keepdim=True)


def avg_pad_2d(t: torch.Tensor, pad: Union[int, tuple[int, int, int, int]]) -> torch.Tensor:
    if isinstance(pad, int):
        pad = (pad, pad, pad, pad)
    glob_avg = torch.mean(t, dim=(-2, -1), keepdim=True)
    t = t - glob_avg
    t = torch.nn.functional.pad(t, pad=pad, mode="constant", value=0)
    return t + glob_avg


def avg_pool2d_fast4bigkernel(t, kernel_size: int):
    """
    Implementation of 2D average pooling with a cumulative sum.
    During the performed tests, dhis implementation was 10x faster than torch implementation for kernel size 41, and
    even faster for bigger kernels. However, this implementation is slower tha torch one for small kernels.
    """
    r = kernel_size // 2
    b, c, h, w = t.shape

    x = torch.nn.functional.pad(t, (r, r, r, r))

    impad = torch.zeros(b, c, h + kernel_size, w + kernel_size, device=t.device)
    impad[..., 1:, 1:] = torch.cumsum(torch.cumsum(x, dim=-1), dim=-2)

    br = impad[..., kernel_size:, kernel_size:]
    tr = impad[..., :h, kernel_size:]
    bl = impad[..., kernel_size:, :w]
    tl = impad[..., :h, :w]

    return (br - tr - bl + tl) / kernel_size**2


def center_with_avg_pool_2d(t: torch.Tensor, kernel_size: int) -> torch.Tensor:

    # kernel_size nust be odd so the output has the same size as the input
    assert kernel_size % 2 == 1, "Please provide an odd kernel size."

    glob_avg = torch.mean(t, dim=(-2, -1), keepdim=True)
    local_avg = t - glob_avg  # Bypass average-padding which is replaced by zeros-padding
    # local_mean = torch.nn.functional.avg_pool2d(
    #     local_mean, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
    local_avg = avg_pool2d_fast4bigkernel(local_avg, kernel_size=kernel_size)
    return t - (local_avg + glob_avg)


def center_with_multiple_avg_pool_2d(t: torch.Tensor, kernel_size: int, n_avg_pool: int) -> torch.Tensor:

    # kernel_size nust be odd so the output has the same size as the input
    assert kernel_size % 2 == 1, "Please provide an odd kernel size."

    glob_avg = torch.mean(t, dim=(-2, -1), keepdim=True)
    local_avg = t - glob_avg  # Bypass average-padding which is replaced by zeros-padding
    for i in range(n_avg_pool):
        # local_mean = torch.nn.functional.avg_pool2d(
        #     local_mean, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
        local_avg = avg_pool2d_fast4bigkernel(local_avg, kernel_size=kernel_size)
    return t - (local_avg + glob_avg)


# =============================================================== #
# ----- Definition of normalization functions for trainings ----- #
# =============================================================== #

# --- MERLIN - Emanuele's normalization
def def_norm_denorm_merlin(**norm_params):

    # Default values used by Emanuele for MERLIN on TerraSAR-X Stripmap images
    eps_log = norm_params.get("eps_log", 1e-3)
    m = norm_params.get("m", -1.429329123112601)
    M = norm_params.get("M", 10.089038980848645)

    def nin(x: torch.Tensor) -> torch.Tensor:
        return ampl_to_norm_log_intens_t(x, M, m, eps_log)

    def ntar(x: torch.Tensor) -> torch.Tensor:
        return x

    def denout(x: torch.Tensor) -> torch.Tensor:
        return denorm_log_intens_t(x, M, m)

    return nin, ntar, denout


# --- MERLIN - no normalization
def def_norm_denorm_merlin_nonorm(**norm_params):

    eps_log = norm_params.get("eps_log", 1e-3)

    def nin(x: torch.Tensor) -> torch.Tensor:
        return ampl_to_log_intens_t(x, eps_log)

    def ntar(x: torch.Tensor) -> torch.Tensor:
        return x

    def denout(x: torch.Tensor) -> torch.Tensor:
        return x

    return nin, ntar, denout


# --- MERLIN - normalization and (local) centering
def def_norm_denorm_merlin_centloc(**norm_params):

    # Default values used by Thomas for MERLIN on calibrated Sentinel-1 IW1 images
    eps_log = norm_params.get("eps_log", 1e-3)
    s = norm_params.get("s", None)
    alpha = norm_params.get("alpha", 1.)
    mean_mode = norm_params.get("mean_mode", "multiple_avgpool")

    def nin(x: torch.Tensor) -> torch.Tensor:
        y = ampl_to_log_intens_t(x, eps_log)
        return alpha / s * y

    def ntar(x: torch.Tensor) -> torch.Tensor:
        return x

    def denout(x: torch.Tensor) -> torch.Tensor:
        y = x / alpha * s
        return 0.5 * y

    if mean_mode == "global":

        def nin2(y: torch.Tensor) -> torch.Tensor:
            return center_each_batch_chan(y)

    elif mean_mode == "avgpool":
        kernel_size = norm_params.get("kernel_size", 121)

        def nin2(y: torch.Tensor) -> torch.Tensor:
            return center_with_avg_pool_2d(y, kernel_size=kernel_size)

    elif mean_mode == "multiple_avgpool":
        kernel_size = norm_params.get("kernel_size", 61)
        n_avg_pool = norm_params.get("n_avg_pool", 4)

        def nin2(y: torch.Tensor) -> torch.Tensor:
            return center_with_multiple_avg_pool_2d(y, kernel_size=kernel_size, n_avg_pool=n_avg_pool)

    else:
        raise AssertionError(f"Unknown mean_mode: {mean_mode} [def_new_norm_denorm_merlin].")

    return nin, ntar, denout, nin2


# --- SAR input - normalization and (local) centering
def def_norm_denorm_sar_centloc(**norm_params):

    # Default values used by Thomas for MERLIN on calibrated Sentinel-1 IW1 images
    eps_log = norm_params.get("eps_log", 1e-3)
    s = norm_params.get("s", None)
    alpha = norm_params.get("alpha", 1.)
    mean_mode = norm_params.get("mean_mode", "multiple_avgpool")

    def nin(x: torch.Tensor) -> torch.Tensor:
        y = ampl_to_log_intens_t(x, eps_log)
        return alpha / s * y

    def ntar(x: torch.Tensor) -> torch.Tensor:
        return x

    def denout(x: torch.Tensor) -> torch.Tensor:
        return x

    if mean_mode == "global":

        def nin2(y: torch.Tensor) -> torch.Tensor:
            return center_each_batch_chan(y)

    elif mean_mode == "avgpool":
        kernel_size = norm_params.get("kernel_size", 121)

        def nin2(y: torch.Tensor) -> torch.Tensor:
            return center_with_avg_pool_2d(y, kernel_size=kernel_size)

    elif mean_mode == "multiple_avgpool":
        kernel_size = norm_params.get("kernel_size", 61)
        n_avg_pool = norm_params.get("n_avg_pool", 4)

        def nin2(y: torch.Tensor) -> torch.Tensor:
            return center_with_multiple_avg_pool_2d(y, kernel_size=kernel_size, n_avg_pool=n_avg_pool)

    else:
        raise AssertionError(f"Unknown mean_mode: {mean_mode} [def_new_norm_denorm_merlin].")

    return nin, ntar, denout, nin2


# ************************************************************************ #
#                                                                          #
# ---------- Diverse normalizations to correctly display images ---------- #
#                                                                          #
# ************************************************************************ #

def get_meanstd(arr: np.ndarray, mstd=3.) -> float:
    return float(np.mean(arr)) + mstd * float(np.std(arr))


def minmax_norm(arr: np.ndarray, mini: float = 0., maxi: float = 1.) -> np.ndarray:
    return (arr - mini) / (maxi - mini)


def minmax_denorm(arr: np.ndarray, mini: float = 0., maxi: float = 1.) -> np.ndarray:
    return arr * (maxi - mini) + mini


def minmax_normclip(arr: np.ndarray, mini: float = 0., maxi: float = 1.) -> np.ndarray:
    return np.clip(minmax_norm(arr, mini=mini, maxi=maxi), 0, 1)


def multip_clip(arr: np.ndarray, mult: float = 1.) -> np.ndarray:
    return np.clip(arr * mult, 0., 1.)


NORM_CONSTS = {
    "m": -1.429329123112601, "M": 10.089038980848645,
    "d": 0, "D": 1.2,
    "g": 0, "G": 4.5,
    "r": 1.2533141373155001, "R": 0.6551363775620336  # mean & std residual noise
}


def get_norm_consts():
    return NORM_CONSTS


def get_norm_const(key):
    return NORM_CONSTS[key]


def norm_meth_cmap_from_name(
        name: str, do_norm: bool = True, do_clip: bool = True
) -> Union[tuple[str, str], tuple[list[str], list[str]]]:

    # Prefix of the normalization method
    if do_norm and do_clip:
        pref_meth = "normclip_"
    elif do_clip:
        pref_meth = "clip_"
    elif do_norm:
        pref_meth = "norm_"
    else:
        return "", "grey"

    # Ground truth databases
    if name.lower() == "mapbiomasalerta":
        return "", "grey"

    # Miscellaneous
    elif ("histo-" in name) and (("_norm_by_" in name) or ("_whiten_by_" in name)):
        return "clip_-4.4172_4.4172", "grey"  # 0.001% of Gaussian law
    elif name.startswith("th_") or name.startswith("th-"):
        return "", "grey"
    elif "pxloss_" in name:
        return pref_meth + "quant_0.05_0.95", "turbo"
    elif ("log_" in name) and not ("_log_" in name):
        return pref_meth + "1m_1M", "grey"
    elif "spectrum" in name:
        return pref_meth + "quant_0_0.95", "grey"
    elif "autocorr" in name:
        return pref_meth + "-1.07_1.07", "bwr"

    # Fusion of images listed afterward
    elif "delta" in name:
        if "noisy" in name:
            return pref_meth + "d_2D", "turbo"
        else:
            return pref_meth + "d_D", "turbo"

    elif "fus_2unc" in name:
        return pref_meth + "d_D", "turbo"
    elif "gamma" in name:
        return pref_meth + "g_G", "turbo"
    elif ("_norm_by_" in name) and ("abs_" in name):
        return pref_meth + "g_G", "turbo"
    elif "_norm_by_" in name:
        return pref_meth + "-1.96_1.96", "viridis"
    elif "_whiten_by_" in name:
        return pref_meth + "-1.96_1.96", "viridis"

    # Variance and co.
    elif "var_decorr_ponds" in name:
        return [pref_meth + "quant_0_0.975", pref_meth + "quant_0.025_0.975", pref_meth + "0_0.8"], ["plasma", "cividis", "pink"]
    elif "var_decorr" in name:
        return [pref_meth + "quant_0_0.975", pref_meth + "quant_0.025_0.975"], ["plasma", "cividis"]
    elif "var_whit_ponds" in name:
        return [pref_meth + "quant_0_0.975", pref_meth + "-1.96_1.96", pref_meth + "0_1"], ["plasma", "viridis", "pink"]
    elif "var_whit" in name:
        return [pref_meth + "quant_0_0.975", pref_meth + "-1.96_1.96"], ["plasma", "viridis"]
    elif "var_law" in name:
        return pref_meth + "d_D", "plasma"
    elif "covmat_law-" in name:
        return pref_meth + "d_D", "plasma"
    elif "filt" in name:
        return pref_meth + "-1_1", "viridis"
    elif "ad_conf_pred" in name:
        return pref_meth + "g_G", "turbo"

    # Basic cases - SAR
    elif "abs_diff_log_ref" in name:
        return pref_meth + "d_D", "cividis"
    elif "diff_log_ref" in name:
        return pref_meth + "-1D_D", "cividis"
    elif "noisy" in name:
        return pref_meth + "meanstd_3", "grey"
    elif "denoised" in name:
        return pref_meth + "meanstd_3", "grey"

    # Sort basic cases (listed in the end so they don't interfere with previous cases)
    elif "gwn" in name:
        return pref_meth + "-1.96_1.96", "viridis"
    elif "gt" in name:
        return "", "grey"

    raise AssertionError("Unrecognized name '" + name + "' to determine the right normalization method.")


def get_extremums_by_method(
        im: np.ndarray, im4norm: np.ndarray = None, normalization_method: str = None
) -> tuple[float, float]:

    consts_norm = get_norm_consts()
    keys_const = consts_norm.keys()

    if im4norm is None:
        im4norm = im.copy()

    sp_norm_meth = normalization_method.split("_")

    if sp_norm_meth[1] == "quant":
        mini = np.min(im4norm) if float(sp_norm_meth[2]) == 0 else np.quantile(im4norm, float(sp_norm_meth[2]))
        maxi = np.max(im4norm) if float(sp_norm_meth[3]) == 1 else np.quantile(im4norm, float(sp_norm_meth[3]))
    elif sp_norm_meth[1] == "minmax":
        mini = np.min(im4norm)
        maxi = np.max(im4norm)
    elif sp_norm_meth[1] == "meanstd":
        mini = 0
        maxi = get_meanstd(im4norm, float(sp_norm_meth[2]))
    else:
        minmax = [-np.inf, np.inf]
        for i in [1, 2]:
            if sp_norm_meth[i][-1] in keys_const:
                if len(sp_norm_meth[i]) == 1:
                    minmax[i - 1] = consts_norm[sp_norm_meth[i]]
                else:
                    try:
                        minmax[i - 1] = consts_norm[sp_norm_meth[i][-1]] * float(sp_norm_meth[i][:-1])
                    except:
                        str2eval = sp_norm_meth[i]
                        for k in keys_const:
                            str2eval = str2eval.replace(k, str(consts_norm[k]))
                        minmax[i - 1] = eval(str2eval)
            else:
                minmax[i - 1] = float(sp_norm_meth[i])
        mini = minmax[0]
        maxi = minmax[1]

    return mini, maxi


def get_list_custom_cmaps() -> list[str]:
    return ["bwr"]


def get_custom_cmap(name_cmap: str, cmap_length: int = 256) -> np.ndarray:
    arng = ((np.arange(cmap_length) / (cmap_length-1)) - 0.5) * 2
    cust_cmap = np.empty((*arng.shape[:-1], 3))

    if name_cmap == "bwr":
        norm_im = cust_cmap * 2 - 1
        cust_cmap[..., 2] = 1 + np.clip(norm_im, -1, 0)
        cust_cmap[..., 0] = 1 - np.clip(norm_im, 0, 1)
        cust_cmap[..., 1] = 1 + np.clip(norm_im, -1, 0) - np.clip(norm_im, 0, 1)

    else:
        raise "Unknown custom color map name '" + name_cmap + "' for the creation of a custom colormap."

    return cust_cmap


def apply_custom_cmap(norm_im_gray: np.ndarray, name_cmap: str) -> np.ndarray:
    if norm_im_gray.shape[-1] != 1:
        warnings.warn("Please provide a grayscale image to apply a colormap. "
                      "A channel will be added to the shape: " + str(norm_im_gray.shape))
        norm_im_gray = norm_im_gray[..., np.newaxis]

    new_im = np.empty((*norm_im_gray.shape[:-1], 3))

    if name_cmap == "bwr":
        norm_im_gray = norm_im_gray * 2 - 1
        new_im[..., 2] = 1 + np.clip(norm_im_gray[..., 0], -1, 0)
        new_im[..., 0] = 1 - np.clip(norm_im_gray[..., 0], 0, 1)
        new_im[..., 1] = 1 + np.clip(norm_im_gray[..., 0], -1, 0) - np.clip(norm_im_gray[..., 0], 0, 1)

    else:
        raise "Unknown custom color map name '" + name_cmap + "' for the application of a custom colormap."
    return new_im


def cmap2apply_from_name(cmap_name: str) -> tuple[Any, bool]:
    is_custom_cmap = False
    if cmap_name is None:
        cmap2apply = None
    elif cmap_name == "grey":
        cmap2apply = None

    elif cmap_name == "turbo":
        cmap2apply = cv2.COLORMAP_TURBO
    elif cmap_name == "jet":
        cmap2apply = cv2.COLORMAP_JET
    elif cmap_name.lower() == "rainbow":
        cmap2apply = cv2.COLORMAP_RAINBOW

    elif cmap_name == "viridis":
        cmap2apply = cv2.COLORMAP_VIRIDIS
    elif cmap_name == "cividis":
        cmap2apply = cv2.COLORMAP_CIVIDIS
    elif cmap_name == "plasma":
        cmap2apply = cv2.COLORMAP_PLASMA
    elif cmap_name == "inferno":
        cmap2apply = cv2.COLORMAP_INFERNO
    elif cmap_name == "magma":
        cmap2apply = cv2.COLORMAP_MAGMA

    elif cmap_name == "pink":
        cmap2apply = cv2.COLORMAP_PINK
    elif cmap_name == "spring":
        cmap2apply = cv2.COLORMAP_SPRING
    elif cmap_name == "cool":
        cmap2apply = cv2.COLORMAP_COOL
    elif cmap_name == "hot":
        cmap2apply = cv2.COLORMAP_HOT
    elif cmap_name == "ocean":
        cmap2apply = cv2.COLORMAP_OCEAN

    elif cmap_name == "twilight":
        cmap2apply = cv2.COLORMAP_TWILIGHT
    elif cmap_name == "twilight_shifted":
        cmap2apply = cv2.COLORMAP_TWILIGHT_SHIFTED
    elif cmap_name.lower() == "hsv":
        cmap2apply = cv2.COLORMAP_HSV

    elif cmap_name in get_list_custom_cmaps():
        is_custom_cmap = True
        cmap2apply = cmap_name
    else:
        raise "Unknown color map '" + cmap_name + "' for normalization."

    return cmap2apply, is_custom_cmap


def normalize_by_method(
        im: np.ndarray, im4norm: np.ndarray = None, normalization_method: str = None, out_cmap: str = "grey"
) -> np.ndarray:

    im_norm = im.copy()
    new_mini, new_maxi = None, None

    if isinstance(normalization_method, list):
        # Repeating the last normalization method for all the last channels
        if isinstance(out_cmap, str):
            out_cmap = [out_cmap for _ in range(im.shape[-1])]
        else:
            # Repeating the last colormap method for all the last channels
            out_cmap += [out_cmap[-1]] * (im.shape[-1] - len(out_cmap))

        if im4norm is None:
            l_im_norm = []
            for c in range(len(normalization_method)):
                if (c == len(normalization_method) - 1) and (c < im.shape[-1] - 1):
                    # Normalizing all the last channels at the same time
                    l_im_norm.append(normalize_by_method(
                        im[..., c:], normalization_method=normalization_method[c], out_cmap=out_cmap[c]))
                else:
                    l_im_norm.append(normalize_by_method(
                        im[..., c:c+1], normalization_method=normalization_method[c], out_cmap=out_cmap[c]))
            im_norm = np.concatenate(l_im_norm, axis=-1)

        else:
            l_im_norm = []
            for c in range(len(normalization_method)):
                if (c == len(normalization_method) - 1) and (c < im.shape[-1] - 1):
                    # Normalizing all the last channels at the same time
                    if im4norm.shape[-1] == 1:
                        l_im_norm.append(normalize_by_method(
                            im[..., c:], im4norm=im4norm,
                            normalization_method=normalization_method[c], out_cmap=out_cmap[c]))
                    else:
                        l_im_norm.append(normalize_by_method(
                            im[..., c:], im4norm=im4norm[..., c:],
                            normalization_method=normalization_method[c], out_cmap=out_cmap[c]))
                else:
                    if im4norm.shape[-1] == 1:
                        l_im_norm.append(normalize_by_method(
                            im[..., c:c+1], im4norm=im4norm,
                            normalization_method=normalization_method[c], out_cmap=out_cmap[c]))
                    else:
                        l_im_norm.append(normalize_by_method(
                            im[..., c:c+1], im4norm=im4norm[..., c:c+1],
                            normalization_method=normalization_method[c], out_cmap=out_cmap[c]))
            im_norm = np.concatenate(l_im_norm, axis=-1)

        return im_norm

    elif (normalization_method is not None) and (normalization_method != ""):

        if im4norm is None:
            im4norm = im.copy()

        sp_norm_meth = normalization_method.split("_")
        method = sp_norm_meth[0]

        if method in ["clip", "norm", "normclip", "clipnorm"]:
            mini, maxi = get_extremums_by_method(im, im4norm=im4norm, normalization_method=normalization_method)

            if method == "clip":
                im_norm = np.clip(im, mini, maxi)
                new_mini, new_maxi = mini, maxi
            elif method == "norm":
                im_norm = minmax_norm(im, mini=mini, maxi=maxi)
                new_mini, new_maxi = 0, 1
            else:
                im_norm = minmax_normclip(im, mini=mini, maxi=maxi)
                new_mini, new_maxi = 0, 1

        elif method == "multip":
            im_norm = multip_clip(im, float(sp_norm_meth[1]))

        else:
            im_norm = im.copy()
            warnings.warn("[normalize_by_method] Unknown normalization method " + method + " from "
                          + normalization_method + ".")

    if isinstance(out_cmap, list):
        # Repeating the last colormap method for all the last channels
        out_cmap += [out_cmap[-1]] * (im.shape[-1] - len(out_cmap))
        l_cm2ap_iscust = [cmap2apply_from_name(cm) for cm in out_cmap]
        l_cmap2apply = [cm2ap_iscust[0] for cm2ap_iscust in l_cm2ap_iscust]
        l_is_custom_cmap = [cm2ap_iscust[1] for cm2ap_iscust in l_cm2ap_iscust]
        npdtype = im_norm.dtype

        lc_im_norm = []
        for c in range(len(out_cmap)):
            if l_cmap2apply[c] is None:
                lc_im_norm.append(im_norm[..., c:c+1])
            else:
                if new_mini is None:
                    new_mini = np.min(im_norm[..., c])
                if new_maxi is None:
                    new_maxi = np.max(im_norm[..., c])

                if not l_is_custom_cmap[c]:
                    lc_im_norm.append(np.stack([minmax_denorm(
                        cv2.applyColorMap(
                            (minmax_norm(im_norm[d, ..., c], new_mini, new_maxi) * 255).astype(np.uint8), l_cmap2apply[c]
                        ).astype(npdtype) / 255, new_mini, new_maxi)
                        for d in range(im_norm.shape[0])], axis=0))
                else:
                    lc_im_norm.append(minmax_denorm(
                        apply_custom_cmap(minmax_norm(im_norm[..., c:c+1], new_mini, new_maxi), l_cmap2apply[c])
                        .astype(npdtype), new_mini, new_maxi))
        im_norm = np.concatenate(lc_im_norm, axis=-1)

    else:
        cmap2apply, is_custom_cmap = cmap2apply_from_name(out_cmap)
        if cmap2apply is not None:
            npdtype = im_norm.dtype
            if new_mini is None:
                new_mini = np.min(im_norm)
            if new_maxi is None:
                new_maxi = np.max(im_norm)

            if not is_custom_cmap:
                im_norm = np.stack(
                    [np.concatenate(
                        [minmax_denorm(
                            cv2.applyColorMap(
                                (minmax_norm(im_norm[d, ..., c], new_mini, new_maxi) * 255).astype(np.uint8), cmap2apply
                            ).astype(npdtype) / 255, new_mini, new_maxi)
                            for c in range(im_norm.shape[3])], axis=2)
                        for d in range(im_norm.shape[0])], axis=0)
            else:
                im_norm = np.concatenate(
                    [minmax_denorm(apply_custom_cmap(minmax_norm(im_norm[..., c:c+1], new_mini, new_maxi), cmap2apply)
                                   .astype(npdtype), new_mini, new_maxi)
                     for c in range(im_norm.shape[3])], axis=3)

    return im_norm
