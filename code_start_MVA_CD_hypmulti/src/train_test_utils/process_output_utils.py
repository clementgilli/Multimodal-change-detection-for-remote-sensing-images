# General

# Imaging
import torch

# Local


# ============================================= #
# ----- MERLIN output processing function ----- #
# ============================================= #

def proc4merlin(output: torch.Tensor, **kwargs) -> torch.Tensor:
    hat_R = torch.exp(output)
    return torch.sqrt(hat_R)

def proc4merlin_minus_eps_log(output: torch.Tensor, eps_log: float, **kwargs) -> torch.Tensor:
    hat_R = torch.exp(output) - eps_log
    return torch.sqrt(hat_R)


# ======================================================== #
# ----- MERLIN uncertainty quantification processing ----- #
# ======================================================== #

# --- Estimation of E[ | log hat_r(a) - log hat_r(b) | ]
def proc4expect_abs_diff(output: torch.Tensor, **kwargs) -> torch.Tensor:
    return torch.nn.Softplus(beta=1)(output)

# --- Estimation of VAR[ log hat_r(a) - log hat_r(b) ]
def var_diff_log_ref(output: torch.Tensor, **kwargs) -> torch.Tensor:
    return torch.nn.Softplus(beta=1)(output)
