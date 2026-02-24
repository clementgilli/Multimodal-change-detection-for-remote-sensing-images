# General
import os
from typing import Callable

# Imaging
import numpy as np
import cv2

# Local
from src.processing import normalization


def save_data_with_ext(
        im, path, ext, save_info_npy=False, im4norm=None, norm_method=None, m255=True, save_cmap="grey", isrgb=False):

    if ext in [".npy", ".np"]:
        np.save(os.path.splitext(path)[0], im)
        if save_info_npy:
            f = open(os.path.splitext(path)[0] + ".inf", "w")
            f.write(" ".join([str(shp) for shp in im.shape]) + "\n")
            f.write("-type " + str(im.dtype))
    elif ext == ".png":
        store_data_png(im, path, im4norm=im4norm, norm_method=norm_method, m255=m255, save_cmap=save_cmap, isrgb=isrgb)
    else:
        raise AssertionError("Unknown extension " + ext + " to save " + path)


def store_data_png(im, filepath, im4norm=None, norm_method=None, m255=True, save_cmap="grey", isrgb=False):

    im = normalization.normalize_by_method(im, im4norm=im4norm,normalization_method=norm_method, out_cmap=save_cmap)

    if m255:
        im = im*255

    filepath = os.path.splitext(filepath)[0]
    fpath = filepath.split("/")
    fold_path = "/".join(fpath[:-1])
    file_name = fpath[-1].split("_")
    file_name = [file_name[0], "_".join(file_name[1:])]
    if file_name[1] != "":
        file_name[1] = "_" + file_name[1]
    file_name[1] += ".png"

    shp = im.shape
    if len(shp) == 2:
        D, H, W, C = 1, shp[0], shp[1], 1
        im = im[np.newaxis, ..., np.newaxis]
    elif len(shp) == 3:
        D, H, W, C = 1, shp[0], shp[1], shp[2]
        im = im[np.newaxis, ...]
    else:
        D, H, W, C = shp[0], shp[1], shp[2], shp[3]

    if (D == 1) and ((C == 1) or ((C == 3) and isrgb)):  # 1 date, 1 channel
        save_path_gen: Callable[[int, int], str] = lambda ch, da: filepath + ".png"
    elif D == 1:  # 1 date, multi channels
        save_path_gen: Callable[[int, int], str] = lambda ch, da: os.path.join(
            fold_path, file_name[0] + "_chan" + str(ch) + file_name[1])
    elif (C == 1) or ((C == 3) and isrgb):  # multi dates, 1 channel
        save_path_gen: Callable[[int, int], str] = lambda ch, da: os.path.join(
            fold_path, file_name[0] + "_date" + str(da) + file_name[1])
    else:  # multi dates, multi channels
        save_path_gen: Callable[[int, int], str] = lambda ch, da: os.path.join(
            fold_path, file_name[0] + "_date" + str(da) + "_chan" + str(ch) + file_name[1])

    for d in range(D):
        if not (isrgb or save_cmap != "grey"):
            for c in range(C):
                cv2.imwrite(save_path_gen(c, d), im[d, :, :, c], [cv2.IMWRITE_PNG_COMPRESSION, 6])

        else:
            assert im.shape[3] % 3 == 0, "Image of shape " + str(im.shape) + " can not be a stack of RGB."
            for c in range(0, C, 3):
                cv2.imwrite(save_path_gen(c//3, d), im[d, :, :, c:c+3][..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 6])
