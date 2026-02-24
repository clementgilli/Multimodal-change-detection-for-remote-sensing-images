# General

# Imaging
import numpy as np
import torch

# Local
from src.train_test_utils import process_output_utils


# ================================ #
# ----- MERLIN loss function ----- #
# ================================ #

def pxwise_merlin_loss(output: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    if not kwargs.get("isproc", True):
        if kwargs.get("minus_eps_log", False):
            hat_R = torch.square(process_output_utils.proc4merlin_minus_eps_log(output, **kwargs))
        else:
            hat_R = torch.square(process_output_utils.proc4merlin(output, **kwargs))
    else:
        hat_R = torch.square(output)
    log_hat_R = torch.log(hat_R)
    b_square = torch.square(target)
    return 0.5 * log_hat_R + b_square / hat_R


def merlin_loss(output: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    crp = kwargs.get("szcroploss", 0)
    # b, _, h, w = output.shape
    # return 1 / b * torch.sum(pxwise_merlin_loss(output, target, **kwargs)[..., crp:h-crp, crp:w-crp])
    h, w = output.shape[-2:]
    return torch.mean(pxwise_merlin_loss(output, target, **kwargs)[..., crp:h-crp, crp:w-crp])


# ============================== #
# ----- Mean squared error ----- #
# ============================== #

def pxwise_mse(output: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    if not kwargs.get("isproc", True):
        if kwargs.get("train_mode", "") == "expect_abs_diff_log_ref":
            output = process_output_utils.proc4expect_abs_diff(output, **kwargs)
    return torch.square(output - target)


def mse(output: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    crp = kwargs.get("szcroploss", 0)
    h, w = output.shape[-2:]
    return torch.mean(pxwise_mse(output, target, **kwargs)[..., crp:h-crp, crp:w-crp])


def pxwise_mse_abs_target(output: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    if not kwargs.get("isproc", True):
        if kwargs.get("train_mode", "") == "expect_abs_diff_log_ref":
            output = process_output_utils.proc4expect_abs_diff(output, **kwargs)
    return torch.square(output - torch.abs(target))


def mse_abs_target(output: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    crp = kwargs.get("szcroploss", 0)
    h, w = output.shape[-2:]
    return torch.mean(pxwise_mse_abs_target(output, target, **kwargs)[..., crp:h-crp, crp:w-crp])


# ===================================================================== #
# ----- Negative log-likelihood for pixelwise variance estimation ----- #
# ===================================================================== #

def pxwise_neg_log_likelihood_normal(output: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    # La formule de base est la suivante mais on peut retirer la constante log(sqrt(2.pi))
    # return torch.mean(
    #     0.5 * torch.log(2*np.pi * torch.square(sigma)) + torch.square(target / sigma) / 2
    # )
    if not kwargs.get("isproc", True):
        if kwargs.get("train_mode", "") == "var_diff_log_ref":
            output = process_output_utils.var_diff_log_ref(output, **kwargs)

    return 0.5 * (np.log(2*np.pi) + torch.log(output) + torch.square(target) / output)


def neg_log_likelihood_normal(output: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    crp = kwargs.get("szcroploss", 0)
    b, _, h, w = output.shape
    return 1 / b * torch.sum(pxwise_neg_log_likelihood_normal(output, target, **kwargs)[..., crp:h-crp, crp:w-crp])
