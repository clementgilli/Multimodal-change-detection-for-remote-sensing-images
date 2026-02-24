# General
from typing import Union

# Imaging
import numpy as np
import torch
from scipy import signal

# Local


def symetrisation_patch_gen(ima):
    # print("symetrisation_patch_gen in progress...")

    assert (len(ima.shape) == 3 and ima.shape[2] == 2), (
        "'symetrisation_patch_gen' requires a 3D image where the 3rd axis stands for real and imaginary parts.")
    S = np.fft.fftshift(np.fft.fft2(ima[:, :, 0]+1j*ima[:, :, 1]))

    p = np.zeros((S.shape[0]))  # azimut (ncol)
    for i in range(S.shape[0]):
        p[i] = np.mean(np.abs(S[i, :]))
    sp = p[::-1]

    # correlation
    c = np.real(np.fft.ifft(np.fft.fft(p)*np.conjugate(np.fft.fft(sp))))
    d1 = np.unravel_index(c.argmax(), p.shape[0])
    d1 = d1[0]

    shift_az_1 = int(round(-(d1-1)/2)) % p.shape[0]+int(p.shape[0]/2)
    p2_1 = np.roll(p, shift_az_1)

    shift_az_2 = int(round(-(d1-1-p.shape[0])/2)) % p.shape[0]+int(p.shape[0]/2)
    p2_2 = np.roll(p, shift_az_2)

    window = signal.gaussian(p.shape[0], std=0.2*p.shape[0])
    test_1 = np.sum(window*p2_1)
    test_2 = np.sum(window*p2_2)

    # make sure the spectrum is symetrized and zeo-Doppler centered
    if test_1 >= test_2:
        p2 = p2_1
        shift_az = shift_az_1/p.shape[0]
    else:
        p2 = p2_2
        shift_az = shift_az_2/p.shape[0]
    S2 = np.roll(S, int(shift_az*p.shape[0]), axis=0)

    q = np.zeros((S.shape[1]))  # range (nlin)
    for j in range(S.shape[1]):
        q[j] = np.mean(np.abs(S[:, j]))
    sq = q[::-1]

    # correlation
    cq = np.real(np.fft.ifft(np.fft.fft(q)*np.conjugate(np.fft.fft(sq))))
    d2 = np.unravel_index(cq.argmax(), q.shape[0])
    d2 = d2[0]

    shift_range_1 = int(round(-(d2-1)/2)) % q.shape[0]+int(q.shape[0]/2)
    q2_1 = np.roll(q, shift_range_1)

    shift_range_2 = int(round(-(d2-1-q.shape[0])/2)) % q.shape[0]+int(q.shape[0]/2)
    q2_2 = np.roll(q, shift_range_2)

    window_r = signal.gaussian(q.shape[0], std=0.2*q.shape[0])
    test_1 = np.sum(window_r*q2_1)
    test_2 = np.sum(window_r*q2_2)

    # make sure the spectrum is symetrized and zeo-Doppler centered
    if test_1 >= test_2:
        q2 = q2_1
        shift_range = shift_range_1/q.shape[0]
    else:
        q2 = q2_2
        shift_range = shift_range_2/q.shape[0]
    Sf = np.roll(S2, int(shift_range*q.shape[0]), axis=1)

    ima2 = np.fft.ifft2(np.fft.ifftshift(Sf))
    ima2 = np.stack((np.real(ima2), np.imag(ima2)), axis=2)
    return ima2.astype(ima.dtype)


# ============================================================= #
# ----- Managing the shape of np.ndarray and torch.Tensor ----- #
# ============================================================= #

def get_bc_to_manage_shape(
        shp: list[int], nc: int = None, more_batch_than_chan: bool = None
) -> tuple[int, int]:
    """
    :param shp: shape for which we want to locate the batch and the channel axes.
    :param nc: number of channels to be searched if it is provided.
    :param more_batch_than_chan: if provided, defines whether there should be a bigger batch-size or a bigger number of
            channels.
            This parameter is ignored is nc is not None.
            If more_batch_than_chan is None, then get_bc_to_manage_shape keeps the order of the axes.
                For example, [1 x 256 x 256 x 3] will output [1 x 3 x 256 x 256]. But also [3 x 256 x 256 x 1] will
                output [3 x 1 x 256 x 256].

    :return: Returning the indices of both the batch axis b and the channel axis c.
    """

    shp = list(shp)  # Ensure that it is a list
    nd = len(shp)

    # Two dimensions: H and
    if nd == 2:
        b, c = None, None

    # Three dimensions H, W and either B or C
    elif nd == 3:

        i = np.argmin(shp).item()
        if shp[i] == nc:  # i was the index of C
            b, c = None, i
        elif nc is not None:  # i was the index of B
            b, c = i, None
        elif more_batch_than_chan:
            b, c = i, None
        else:  # Default: i was the index of C
            b, c = None, i

    # Four dimensions (or more)
    else:
        idx_sort_shp = np.argsort(shp)  # Is not reliable if B == C

        if nc is None:
            i = np.argmin(shp).item()  # Index of lowest between B and C
            modshp = shp.copy()
            modshp[i] = np.inf
            j = np.argmin(modshp).item()  # Index of highest between B and C

            if idx_sort_shp[0] == idx_sort_shp[1]:  # Here, j in the index of C because i < j
                b, c = i, j

            elif more_batch_than_chan is None:  # Keeping the initial order of B and C
                if idx_sort_shp[0] < idx_sort_shp[1]:  # Here, B < C, so j is the index of C
                    b, c = i, j
                else:  # Here, B > C, so j is the index of B
                    b, c = j, i

            elif more_batch_than_chan:  # Here we want B > C, so j is the index of B
                b, c = j, i

            else:  # Here we want B < C, so j is the index of C
                b, c = i, j

        else:  # Ignoring more_batch_than_chan
            assert nc in shp, f"Invalid number of channel nc={nc} for the shape {shp}."

            if (nc in np.array(shp)[idx_sort_shp[:2]]) and (shp[idx_sort_shp[0]] == shp[idx_sort_shp[1]]):  # B == C
                b = np.argmin(shp).item()  # Lowest index between the index of B and C (index of B)
                modshp = shp.copy()
                modshp[b] = np.inf
                c = np.argmin(modshp).item()  # Highest index between the index of B and C (index of C)

            else:
                c = shp.index(nc)  # Index of C
                modshp = shp.copy()
                modshp[c] = np.inf
                b = np.argmin(modshp).item()  # Index of B

    return b, c


def manage_shape_npy_from_bc(arr: np.ndarray, b: int, c: int) -> np.ndarray:
    """
    :param arr: array to be modified.
    :param b: index of the batch axis B.
    :param c: index of the channel axis C.

    :return: Returning the array arr after swapping axis to have a [B x H x W x C] array.
            With: B the batch-size, C the number of channels, H the height and W the width.
    """
    shp = arr.shape
    nd = len(shp)

    if nd == 2:  # b and c are None
        return arr[np.newaxis, ..., np.newaxis]

    elif nd == 3:  # b or c is None
        hw = [d for d in range(nd) if d not in [b, c]]
        if b is None:
            return arr.transpose(c, *hw)[np.newaxis, ...]
        else:
            return arr.transpose(b, *hw)[:, np.newaxis, ...]

    else:  # Neither b nor c is None
        hw = [d for d in range(nd) if d not in [b, c]]
        return arr.transpose(b, c, *hw)


def manage_shape_npy(arr: np.ndarray, nc: int = None, more_batch_than_chan: bool = None) -> np.ndarray:
    """
    :param arr: array to be modified.
    :param nc: number of channels to be searched if it is provided.
    :param more_batch_than_chan: if provided, defines whether there should be a bigger batch-size or a bigger number of
            channels.
            This parameter is ignored is nc is not None.
            If more_batch_than_chan is None, then manage_shape_npy keeps the order of the dimensions.
                For example, [1 x 3 x 256 x 256] will output [1 x 256 x 256 x 3]. But also [3 x 1 x 256 x 256] will
                output [3 x 256 x 256 x 1].

    :return: Returning the array arr after swapping axis to have a [B x H x W x C] array.
            With: B the batch-size, C the number of channels, H the height and W the width.
    """

    shp = list(arr.shape)
    b, c = get_bc_to_manage_shape(shp, nc=nc, more_batch_than_chan=more_batch_than_chan)
    return manage_shape_npy_from_bc(arr, b, c)


def manage_shape_torch_from_bc(t: torch.Tensor, b: int, c: int) -> torch.Tensor:
    """
    :param t: tensor to be modified.
    :param b: index of the batch axis B.
    :param c: index of the channel axis C.

    :return: Returning the tensor t after swapping axis to have a [B x C x H x W] tensor.
            With: B the batch-size, C the number of channels, H the height and W the width.
    """
    shp = t.shape
    nd = len(shp)

    if nd == 2:  # b and c are None
        return t[torch.newaxis, torch.newaxis, ...]

    elif nd == 3:  # b or c is None
        hw = [d for d in range(nd) if d not in [b, c]]
        if b is None:
            return t.permute(c, *hw)[torch.newaxis, ...]
        else:
            return t.permute(b, *hw)[:, torch.newaxis, ...]

    else:  # Neither b nor c is None
        hw = [d for d in range(nd) if d not in [b, c]]
        return t.permute(b, c, *hw)


def manage_shape_torch(t: torch.Tensor, nc: int = None, more_batch_than_chan: bool = None) -> torch.Tensor:
    """
    :param t: tensor to be modified.
    :param nc: number of channels to be searched if it is provided.
    :param more_batch_than_chan: if provided, defines whether there should be a bigger batch-size or a bigger number of
            channels.
            This parameter is ignored is nc is not None.
            If more_batch_than_chan is None, then manage_shape_torch keeps the order of the dimensions.
                For example, [1 x 256 x 256 x 3] will output [1 x 3 x 256 x 256]. But also [3 x 256 x 256 x 1] will
                output [3 x 1 x 256 x 256].

    :return: Returning the tensor t after swapping axis to have a [B x C x H x W] tensor.
            With: B the batch-size, C the number of channels, H the height and W the width.
    """

    shp = list(t.shape)
    b, c = get_bc_to_manage_shape(shp, nc=nc, more_batch_than_chan=more_batch_than_chan)
    return manage_shape_torch_from_bc(t, b, c)
