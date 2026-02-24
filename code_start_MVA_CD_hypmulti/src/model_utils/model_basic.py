# General
from collections import OrderedDict
from typing_extensions import override

# Imaging
import torch

# Local
from src.model_utils import general_model
from src.model_utils import archi_unet_merlin, archi_NAFNet


class Model(general_model.Model):

    def __init__(
            self, height: int = 256, width: int = 256,
            norm_params: dict = None,
            train_mode: str = "merlin",
            args4loss: dict = None,
            data_aug_intar: dict = None, data_aug_in: dict = None, data_aug_tar: dict = None,
            device: str = ("cuda:0" if torch.cuda.is_available() else "cpu"),

            mode_body : str = "unet", args4body: dict = None,
            # **kwargs_general
    ):
        super().__init__(
            height=height, width=width,
            train_mode=train_mode, args4loss=args4loss, norm_params=norm_params,
            data_aug_intar=data_aug_intar, data_aug_in=data_aug_in, data_aug_tar=data_aug_tar,
            device=device
            # **kwargs_general
        )

        # =============== #
        # Body parameters #
        # =============== #
        self.mode_body = mode_body
        self.args4body = args4body if (args4body is not None) else {}

        # ================ #
        # ----- BODY ----- #
        # ================ #
        if self.mode_body.lower() == "nafnet":  # NAFNet
            nc_out_body = self.args4body.get("nc_out_body", 32)
            self.body = archi_NAFNet.NAFNet(in_channels=self.nb_in_chan, out_channels=nc_out_body, **self.args4body)
            self.head = torch.nn.Sequential(OrderedDict(
                [("linear_end_head", torch.nn.Conv2d(
                    in_channels=nc_out_body, out_channels=self.nb_out_chan, kernel_size=(3, 3), stride=(1, 1),
                    padding='same'))]
            ))
        else:  # UNet MERLIN
            nc_out_body = self.args4body.get("nc_out_body", 1)
            self.body = archi_unet_merlin.UNetMERLIN(nb_in_channels=self.nb_in_chan, nb_out_channels=nc_out_body)
            self.head = None

        # ================ #
        # ----- HEAD ----- #
        # ================ #


    @override
    def forward_archi(self, x):

        # ================ #
        # ----- TAIL ----- #
        # ================ #

        # ================ #
        # ----- BODY ----- #
        # ================ #
        n = self.body(x)  # Not residual learning

        # ================ #
        # ----- HEAD ----- #
        # ================ #
        if self.head is not None:
            n = self.head(n)

        # return x - n
        return n


    def get_archi_params(self):
        return {  # Must be the arguments of __init__
            "mode_body": self.mode_body,
            "args4body": self.args4body,
        }
