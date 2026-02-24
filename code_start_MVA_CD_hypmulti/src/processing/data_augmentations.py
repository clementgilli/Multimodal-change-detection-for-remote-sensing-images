# General
import numbers
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt

# Imaging
import numpy as np
import torch
import torchvision

# Local


def apply_augmentations(t: torch.Tensor, daug: dict, rand_affine_transformer=None, pad_mod: str="constant",
                        show_diff: bool=False) -> torch.Tensor:
    tinit = None
    if show_diff:
        tinit = t.detach().cpu().numpy().copy()

    l_daugs = daug.keys()

    transfo_done = False
    k_d_transfo = empty_dict4transfo().keys()

    for k in l_daugs:

        if (k in k_d_transfo) or k.startswith("rota") or k.startswith("transl") and (not transfo_done):
            t = rand_affine_transformer(t)
            transfo_done = True

        elif k.startswith("noise"):
            t += (torch.randn(t.shape) if "gauss" in k else (torch.rand(t.shape)-0.5)*2) * daug[k]

        elif k.startswith("offset") or k.startswith("add"):
            t += (torch.randn(1) if "gauss" in k else (torch.rand(1)-0.5)*2) * daug[k]

        elif k.startswith("fact") or k.startswith("mult"):
            t *= (torch.randn(1) if "gauss" in k else (torch.rand(1)-0.5)*2) * daug[k]

        # elif k.startswith("px_transl"):
        #     t = tensor_manag.translat_tensor(
        #         t, tuple(np.random.randint(-daug[k], daug[k], size=(2,))), padding_mode=pad_mod)

    if show_diff:
        plt.figure(dpi=200)
        plt.subplot(2, 3, 1), plt.title("init"), plt.imshow(tinit[0, 0, ...], vmin=0, vmax=500, cmap="grey")
        plt.subplot(2, 3, 2), plt.title("after daug")
        plt.imshow(t[0, 0, ...].detach().cpu().numpy(), vmin=0, vmax=500, cmap="grey")
        plt.subplot(2, 3, 3), plt.title("after daug")
        plt.imshow(np.abs(tinit[0, 0, ...] - t[0, 0, ...].detach().cpu().numpy()), vmin=0, vmax=100, cmap="grey")

        plt.subplot(2, 3, 4), plt.title("init"), plt.imshow(tinit[3, 0, ...], vmin=0, vmax=500, cmap="grey")
        plt.subplot(2, 3, 5), plt.title("after daug")
        plt.imshow(t[3, 0, ...].detach().cpu().numpy(), vmin=0, vmax=500, cmap="grey")
        plt.subplot(2, 3, 6), plt.title("after daug")
        plt.imshow(np.abs(tinit[3, 0, ...] - t[3, 0, ...].detach().cpu().numpy()), vmin=0, vmax=100, cmap="grey")

        plt.show()

    return t


def empty_dict4transfo() -> dict:
    return dict(
        degrees=(0, 0),
        translate=None,
        scale=None,
        shear=None
    )


def get_random_affine_transformer(daug: dict):

    l_daugs = daug.keys()
    d_transfo = empty_dict4transfo()
    k_d_transfo = d_transfo.keys()
    is_transfo = False

    for k in l_daugs:
        if k in k_d_transfo:
            is_transfo = True
            if isinstance(daug[k], numbers.Number) and (k != "degrees"):
                d_transfo[k] = (daug[k], daug[k])
            else:
                d_transfo[k] = daug[k]

        elif k.startswith("translation"):
            is_transfo = True
            if isinstance(daug[k], numbers.Number):
                d_transfo["translate"] = (daug[k], daug[k])
            else:
                d_transfo["translate"] = daug[k]

        elif k.startswith("rota"):
            is_transfo = True
            d_transfo["degrees"] = daug[k]

    if not is_transfo:
        return None
    else:
        return CustomRandomAffine(
            **d_transfo, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)


# Override RandomAffine to be able to make a non integer translation
class CustomRandomAffine(torchvision.transforms.RandomAffine):

    # Override method
    @staticmethod
    def get_params(
            degrees: List[float],
            translate: Optional[List[float]],
            scale_ranges: Optional[List[float]],
            shears: Optional[List[float]],
            img_size: List[int],
    ) -> Tuple[float, Tuple[int, int], float, Tuple[float, float]]:
        """Get parameters for affine transformation

        Returns:
            params to be passed to the affine transformation
        """
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        if translate is not None:
            max_dx = float(translate[0] * img_size[0])
            max_dy = float(translate[1] * img_size[1])
            tx = torch.empty(1).uniform_(-max_dx, max_dx).item()
            ty = torch.empty(1).uniform_(-max_dy, max_dy).item()
            translations = (tx, ty)
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = float(torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item())
        else:
            scale = 1.0

        shear_x = shear_y = 0.0
        if shears is not None:
            shear_x = float(torch.empty(1).uniform_(shears[0], shears[1]).item())
            if len(shears) == 4:
                shear_y = float(torch.empty(1).uniform_(shears[2], shears[3]).item())

        shear = (shear_x, shear_y)

        return angle, translations, scale, shear
