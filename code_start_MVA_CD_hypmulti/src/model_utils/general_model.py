# General
import os
import warnings
from typing import Union
import json
from tqdm import tqdm

# Imaging
import numpy as np
import torch

# Local
from src.manage_data import save_data
from src.misc import misc_xml
from src.processing import data_augmentations
from src.processing import normalization
from src.train_test_utils import loss_functions
from src.train_test_utils import process_output_utils


class Model(torch.nn.Module):

    def __init__(
            self, height: int = 256, width: int = 256,
            train_mode: str = "merlin", args4loss: dict = None, norm_params: dict = None,
            data_aug_intar: dict = None, data_aug_in: dict = None, data_aug_tar: dict = None,
            device: str = ("cuda:0" if torch.cuda.is_available() else "cpu")
    ):
        super().__init__()

        # ============= #
        # Miscellaneous #
        # ============= #
        self.device = device
        self.verbose_infos_model = True

        # ======================== #
        # Normalization parameters #
        # ======================== #
        self.norm_params = norm_params if norm_params is not None else {}
        self.init_normalization()
        self.norm_in = None
        self.norm_tar = None
        self.denorm_out = None
        self.norm_in_2 = None
        self.post_forward = None  # eg.: residual connexion, defined later
        self.proc_forward = None

        # =============================== #
        # Training mode and loss function #
        # =============================== #
        self.train_mode = train_mode
        self.loss_name = "mse"
        self.proc_forward = None
        self.in_kinds = [["input"]]
        self.tar_kinds = [["target"]]
        self.prefix_out = "output"
        self.nb_in_chan = 1
        self.nb_out_chan = 1
        self.swap_in_tar = False
        self.init_training_mode()
        self.def_norm_in_tar_denorm_out()

        self.loss_function = getattr(loss_functions, self.loss_name)
        try:
            self.pxw_loss_function = getattr(loss_functions, f"pxwise_{self.loss_name}")
        except (NameError, AttributeError) as e:
            warnings.warn("Pixelwise loss function not found. Error raised:\n" + str(e))
            self.pxw_loss_function = None
        self.args4loss = args4loss if args4loss is not None else {}

        # ============ #
        # Images shape #
        # ============ #
        self.height = height
        self.width = width

        # ====================================== #
        # Preparation of input / output / target #
        # ====================================== #
        self.data_aug_in = data_aug_in if (data_aug_in is not None) else {}
        self.rand_affine_transformer_in = data_augmentations.get_random_affine_transformer(self.data_aug_in)
        self.data_aug_tar = data_aug_tar if (data_aug_tar is not None) else {}
        self.rand_affine_transformer_tar = data_augmentations.get_random_affine_transformer(self.data_aug_tar)
        self.data_aug_intar = data_aug_intar if (data_aug_tar is not None) else {}
        self.rand_affine_transformer_intar = data_augmentations.get_random_affine_transformer(self.data_aug_intar)

        self.apply_data_aug_in = None
        self.fct_data_aug_tar = None
        self.fct_data_aug_intar = None
        self.def_apply_data_aug_in()
        self.def_fct_data_aug_tar()
        self.def_fct_data_aug_intar()


    def forward(self, x):
        if self.norm_in_2 is None:
            x = self.norm_in(x)
            x = self.apply_data_aug_in(x)
            y = self.forward_archi(x)

        else:
            x = self.norm_in(x)
            x = self.apply_data_aug_in(x)
            y = self.norm_in_2(x)
            y = self.forward_archi(y)

        if self.post_forward is not None:
            y = self.post_forward(y, x)
        y = self.denorm_out(y)
        return y


    def forward_archi(self, x):
        raise NotImplementedError


    def calc_all_outputs(self, x):
        if self.norm_in_2 is None:
            x = self.norm_in(x)
            x = self.apply_data_aug_in(x)
            y = self.forward_archi_all_outputs(x)

        else:
            x = self.norm_in(x)
            x = self.apply_data_aug_in(x)
            y = self.norm_in_2(x)
            y = self.forward_archi_all_outputs(y)

        if self.post_forward is not None:
            y = self.post_forward(y, x)
        return self.denorm_out(y)


    def forward_archi_all_outputs(self, x):
        raise NotImplementedError


    def init_normalization(self):
        init_norm_dict = {
            "norm_fct_name": None,
            # "apply_log_in": False,
            # "eps_log": 1e-3,
            # "mode_eps_log": "offset",
            # "m": None,
            # "M": None
        }
        for k in init_norm_dict.keys():
            if self.norm_params.get(k, None) is None:
                self.norm_params[k] = init_norm_dict[k]


    def init_training_mode(self):

        if self.train_mode == "merlin":
            self.loss_name = "merlin_loss"
            self.proc_forward = lambda out: process_output_utils.proc4merlin(out, **self.args4loss)

            self.in_kinds = [["noisy_real"]]
            self.tar_kinds = [["noisy_imag"]]
            self.prefix_out = "denoised"
            self.nb_in_chan = 1
            self.nb_out_chan = 1

            self.swap_in_tar = True
            self.norm_params["apply_log_in"] = True

            def post_fwd(out: torch.Tensor, inp: torch.Tensor):
                return inp - out
            self.post_forward = post_fwd

        elif self.train_mode == "var_diff_log_ref":
            self.loss_name = "neg_log_likelihood_normal"
            self.proc_forward = lambda out: process_output_utils.var_diff_log_ref(out, **self.args4loss)

            self.in_kinds = [["noisy_real", "denoised_real"], ["noisy_imag", "denoised_imag"]]
            self.tar_kinds = [["diff_log_ref"]]
            self.prefix_out = "var_diff_log_ref"
            self.nb_in_chan = 2
            self.nb_out_chan = 1

            self.swap_in_tar = False
            self.norm_params["apply_log_in"] = True

        else:
            raise f"Unrecognized train_mode '{self.train_mode}'"


    def get_interest_weights(self):
        raise NotImplementedError


    def def_apply_data_aug_in(self):

        if (self.data_aug_in is None) or (self.data_aug_in == {}):
            if self.verbose_infos_model:
                print("Data-augmentation input --- no data-augmentation.")
            def pre_fwd(x):
                return x

        else:
            if self.verbose_infos_model:
                print("Data-augmentation input --- data-augmentation with parameters:", self.data_aug_in)
            def pre_fwd(x):
                return data_augmentations.apply_augmentations(
                    x, self.data_aug_in, self.rand_affine_transformer_in)

        self.apply_data_aug_in = pre_fwd


    def def_fct_data_aug_tar(self):

        if (self.data_aug_tar is None) or (self.data_aug_tar == {}):
            if self.verbose_infos_model:
                print("Data-augmentation target --- no data-augmentation.")
            def fct_daug_tar(x):
                return x

        else:
            if self.verbose_infos_model:
                print("Data-augmentation target --- data-augmentation with parameters:", self.data_aug_in)
            def fct_daug_tar(x):
                return data_augmentations.apply_augmentations(
                    x, self.data_aug_tar, self.rand_affine_transformer_tar)

        self.fct_data_aug_tar = fct_daug_tar


    def def_fct_data_aug_intar(self):

        if (self.data_aug_intar is None) or (self.data_aug_intar == {}):
            if self.verbose_infos_model:
                print("Data-augmentation input and target --- no data-augmentation.")
            def fct_daug_intar(batch):
                return batch

        else:
            if self.verbose_infos_model:
                print("Data-augmentation input and target --- data-augmentation with parameters:", self.data_aug_intar)
            def fct_daug_intar(batch):
                dx = batch[0].shape[1]
                batch = data_augmentations.apply_augmentations(
                    torch.cat(batch, dim=1), self.data_aug_intar,
                    self.rand_affine_transformer_intar)
                return batch[:, :dx, ...], batch[:, dx:, ...]

        self.fct_data_aug_intar = fct_daug_intar


    def def_norm_in_tar_denorm_out(self):

        if self.norm_params.get("norm_fct_name") is not None:
            if self.verbose_infos_model:
                print("Normalization input and output --- custom normalization function "
                      f"'{self.norm_params.get('norm_fct_name')}'.")
            norm_fcts = getattr(normalization, self.norm_params.get("norm_fct_name"))(**self.norm_params)
            self.norm_in = norm_fcts[0]
            self.norm_tar = norm_fcts[1]
            self.denorm_out = norm_fcts[2]
            if len(norm_fcts) > 3:
                self.norm_in_2 = norm_fcts[3]

        else:

            if self.verbose_infos_model:
                print("Normalization target --- no normalization.")
            def ntar(x):
                return x

            apply_log_in = self.norm_params.get("apply_log_in", False)
            eps_log = self.norm_params.get("eps_log", 1e-3)
            mode_eps_log = self.norm_params.get("mode_eps_log", "offset")
            m = self.norm_params.get("m", None)
            M = self.norm_params.get("M", None)

            if not apply_log_in:

                if (M is None) and (m is None):
                    if self.verbose_infos_model:
                        print(f"Normalization input --- no normalization.")
                    def nin(x):
                        return x
                    def denout(x):
                        return x

                elif M is None:
                    if self.verbose_infos_model:
                        print(f"Normalization input --- offset m={m}.")
                    def nin(x):
                        return normalization.ofst_tensor(x, m)
                    def denout(x):
                        return normalization.deofst_tensor(x, m)

                elif m is None:
                    if self.verbose_infos_model:
                        print(f"Normalization input --- dilatation M={M}.")
                    def nin(x):
                        return normalization.dilat_tensor(x, M)
                    def denout(x):
                        return normalization.dedilat_tensor(x, M)

                else:
                    if self.verbose_infos_model:
                        print(f"Normalization input --- normalization (m={m} and M={M}).")
                    def nin(x):
                        return normalization.norm_tensor(x, M, m)
                    def denout(x):
                        return normalization.denorm_tensor(x, M, m)

            else:
                assert mode_eps_log in ["offset", "clamp", "clip"], (
                    "Please provide 'offset' or 'clamp' for the way to apply epsilon in the logarithm (mode_eps_log).")

                if (M is None) and (m is None):
                    if self.verbose_infos_model:
                        print(f"Normalization input --- log-intensity computation with '{mode_eps_log}' mode to apply "
                              f"epsilon inside the logarithm.")
                    fctnorm = (normalization.ampl_to_log_intens_t if (mode_eps_log == "offset")
                               else normalization.ampl_to_log_intens_clip_eps_t)
                    def nin(x):
                        return fctnorm(x, eps_log)
                    def denout(x):
                        return x

                elif M is None:
                    if self.verbose_infos_model:
                        print(f"Normalization input --- normalization without dilatation in log-intensity with "
                              f"'{mode_eps_log}' mode to apply epsilon inside the logarithm.")
                    fctnorm = (normalization.ampl_to_ofst_log_intens_t if (mode_eps_log == "offset")
                               else normalization.ampl_to_ofst_log_intens_clip_eps_t)
                    def nin(x):
                        return fctnorm(x, m, eps_log)
                    def denout(x):
                        return normalization.deofst_log_intens_t(x, m)

                elif m is None:
                    if self.verbose_infos_model:
                        print(f"Normalization input --- normalization without offset in log-intensity with "
                              f"'{mode_eps_log}' mode to apply epsilon inside the logarithm.")
                    fctnorm = (normalization.ampl_to_dilat_log_intens_t if (mode_eps_log == "offset")
                               else normalization.ampl_to_dilat_log_intens_clip_eps_t)
                    def nin(x):
                        return fctnorm(x, M, eps_log)
                    def denout(x):
                        return normalization.deofst_log_intens_t(x, M)

                else:
                    if self.verbose_infos_model:
                        print(f"Normalization input --- normalization in log-intensity with '{mode_eps_log}' "
                              f"mode to apply epsilon inside the logarithm.")
                    fctnorm = (normalization.ampl_to_norm_log_intens_t if (mode_eps_log == "offset")
                               else normalization.ampl_to_norm_log_intens_clip_eps_t)
                    def nin(x):
                        return fctnorm(x, M, m, eps_log)
                    def denout(x):
                        return normalization.denorm_log_intens_t(x, M, m)

            self.norm_in = nin
            self.norm_tar = ntar
            self.denorm_out = denout


    def loss(self, output, target, **kwargs):
        return self.loss_function(output, self.norm_tar(self.fct_data_aug_tar(target)), **kwargs)


    def pxw_loss(self, output, target, **kwargs):
        return self.pxw_loss_function(output, self.norm_tar(target), **kwargs)


    def training_step(self, batch, batch_number):

        x, y = self.fct_data_aug_intar(batch)
        # x = x.to(self.device)
        # y = y.to(self.device)
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        if self.swap_in_tar and (batch_number % 2 == 1):
            out = self.forward(y)
            loss = self.loss(out, x, isproc=False, **self.args4loss)
        else:
            out = self.forward(x)
            loss = self.loss(out, y, isproc=False, **self.args4loss)

        return loss


    def validation_step(self, batch, batch_number, epoch_num, sample_dir, sample_batch_number=0):

        with torch.no_grad():

            x, y = self.fct_data_aug_intar(batch)
            # x = x.to(self.device)
            # y = y.to(self.device)
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            out_tensor, out_proc, outx_proc, outy_proc = [None for _ in range(4)]

            if self.swap_in_tar:
                outx = self.forward(x)
                outy = self.forward(y)
                outx_proc = self.proc_forward(outx)
                outy_proc = self.proc_forward(outy)
                loss = 0.5 * (self.loss(outx_proc, y, isproc=True, **self.args4loss)
                              + self.loss(outy_proc, x, isproc=True, **self.args4loss))

            else:
                out_tensor = self.forward(x)
                loss = self.loss(out_tensor, y, isproc=False, **self.args4loss)

            if batch_number == sample_batch_number:  # Saving a sample at each epoch
                inp = np.moveaxis(x.cpu().numpy()[0, ...], 0, -1)
                tar = np.moveaxis(y.cpu().numpy()[0, ...], 0, -1)

                n_in, cm_in = normalization.norm_meth_cmap_from_name(self.in_kinds[0][0], do_norm=True, do_clip=True)
                n_tar, cm_tar = normalization.norm_meth_cmap_from_name(self.tar_kinds[0][0], do_norm=True, do_clip=True)
                n_out, cm_out = normalization.norm_meth_cmap_from_name(self.prefix_out, do_norm=True, do_clip=True)
                # n_spc, cm_spc = normalization.norm_meth_cmap_from_name("spectrum", do_norm=True, do_clip=True)

                imagename = f"validation_image_patch_{batch_number}_epoch_{epoch_num}.npy"

                if epoch_num == 1: # Saving inputs / targets during the first epoch
                    save_data.save_data_with_ext(inp, os.path.join(sample_dir, "input_" + imagename), ext=".npy")
                    save_data.save_data_with_ext(inp, os.path.join(sample_dir, "input_" + imagename), ext=".png",
                                                 norm_method=n_in, save_cmap=cm_in)

                    save_data.save_data_with_ext(tar, os.path.join(sample_dir, "target_" + imagename), ext=".npy")
                    save_data.save_data_with_ext(tar, os.path.join(sample_dir, "target_" + imagename), ext=".png",
                                                 norm_method=n_tar, save_cmap=cm_tar)

                if (out_proc is None) and (out_tensor is not None):
                    out_proc = self.proc_forward(out_tensor)
                elif (out_proc is None) and (outx_proc is not None):
                    out_proc = torch.sqrt(0.5 * (torch.square(outx_proc) + torch.square(outy_proc)))
                out_proc = np.moveaxis(out_proc.cpu().numpy()[0, ...], 0, -1)

                if out_proc.shape[-1] < 5:
                    save_data.save_data_with_ext(
                        out_proc, os.path.join(sample_dir, f"{self.prefix_out}_{imagename}"), ext=".png",
                        norm_method=n_out, save_cmap=cm_out)
                else:
                    save_data.save_data_with_ext(
                        out_proc, os.path.join(sample_dir, f"{self.prefix_out}_{imagename}"), ext=".npy")

        return loss


    def infer_datas_from_test_dir(
            self, test_dir: str, save_output: bool = True, save_as_png: bool = True, save_dir: str = None
    ) -> Union[np.ndarray, list[np.ndarray]]:
        """
        Cette fonction est écrite un rapidement, n'hésitez pas à la modifier (et 'test_network.py' en conséquence).
        TODO: D'ailleurs, il faudrait idéalement ajouter le calcul de la loss...
        :param test_dir: Fichier '.npy' ou dossier contenant plusieurs fichiers '.npy' à inférer.
        """

        if save_dir is None:
            save_dir = test_dir.split("/")
            save_dir = "/".join(save_dir[:-1]) + f"/{self.prefix_out}_{save_dir[-1]}"

        n_out, cm_out = normalization.norm_meth_cmap_from_name(self.prefix_out, do_norm=True, do_clip=True)

        res = []

        if os.path.isfile(test_dir) and test_dir.endswith(".npy"):

            im_s = np.load(test_dir, mmap_mode="r")
            for i in range(im_s.shape[0]):
                res.append(self.infer_data(im_s[i:i+1, ...]))

            if len(res[0].shape) < 4:
                res = np.stack(res, axis=0)
            else:
                res = np.concatenate(res, axis=0)

            if save_output:
                np.save(save_dir, res)
                if save_as_png:
                    save_data.save_data_with_ext(
                        res, save_dir, ext=".png", norm_method=n_out, save_cmap=cm_out, im4norm=res)

        else:
            list_files = os.listdir(test_dir)

            if save_output:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

            for file in list_files:
                im = np.load(os.path.join(test_dir, file))
                res.append(self.infer_data(im))
                if save_output:
                    np.save(os.path.join(save_dir, f"{self.prefix_out}_{file}"), res)
                    if save_as_png:
                        save_data.save_data_with_ext(
                            res, os.path.join(save_dir, f"{self.prefix_out}_{file}"), ext=".png",
                            norm_method=n_out, save_cmap=cm_out, im4norm=res)

        return res


    def infer_data(self, im, max_sz=1024, stride_ratio=1., sup_border_ratio=0.) -> np.ndarray:
        """
        Cette fonction est écrite un rapidement, n'hésitez pas à la modifier (et 'test_network.py' en conséquence).
        TODO: D'ailleurs, il faudrait idéalement ajouter le calcul de la loss...
        """
        with torch.no_grad():

            ndim = len(im.shape)
            if ndim == 2:
                im = torch.tensor(im[np.newaxis, np.newaxis, ...], device=self.device)
            elif ndim == 3:
                im = torch.tensor(im[np.newaxis, ...], device=self.device).permute(0, 3, 1, 2)
            else:
                im = torch.tensor(im, device=self.device).permute(0, 3, 1, 2)

            B, C, H, W = im.shape

            max_sz = min(max_sz, H, W)
            stride = int(max_sz * stride_ratio)
            crpb = int(max_sz * sup_border_ratio)

            res = torch.empty((B, self.nb_out_chan, H, W), dtype=im.dtype)
            count_image = torch.zeros((1, 1, H, W), dtype=torch.int16)
            # for h in range(0, H-max_sz, stride):
            for h in tqdm(range(0, H-max_sz+1, stride)):
                for w in range(0, W-max_sz+1, stride):
                    patch = im[..., h:h+max_sz, w:w+max_sz].to(self.device)
                    hp, wp = patch.shape[2:]

                    if self.swap_in_tar and (C == 2*self.nb_in_chan):
                        res_p_1 = self.proc_forward(self.forward(patch[:, :C//2, ...]))
                        res_p_2 = self.proc_forward(self.forward(patch[:, C//2:, ...]))
                        res_p = torch.sqrt(0.5 * (res_p_1**2 + res_p_2**2))
                    else:
                        res_p = self.proc_forward(self.forward(patch))

                    res[..., h+crpb:h+hp-crpb, w+crpb:w+wp-crpb] = res_p[..., crpb:hp-crpb, crpb:wp-crpb].detach().cpu()
                    count_image[..., h + crpb:h + hp - crpb, w + crpb:w + wp - crpb] += 1

            res = res / count_image

        if ndim == 2:
            res = res[0, 0, ...].cpu().numpy()
        elif ndim == 3:
            res = res[0, ...].permute(1, 2, 0).cpu().numpy()
        else:
            res = res.permute(0, 2, 3, 1).cpu().numpy()

        return res


    def get_general_model_params(self):
        return {  # Must be the arguments of __init__
            "height": self.height,
            "width": self.width,
            "train_mode": self.train_mode,
            "args4loss": self.args4loss,
            "norm_params": self.norm_params,
            "data_aug_intar": self.data_aug_intar,
            "data_aug_in": self.data_aug_in,
            "data_aug_tar": self.data_aug_tar,
            "device": self.device
        }


    def get_archi_params(self):
        raise NotImplementedError


    def get_all_params(self):
        try:
            return dict(self.get_general_model_params(), **self.get_archi_params())
        except Exception as e:
            return self.get_general_model_params()


    def disp_all_params(self):
        return json.dumps(self.get_all_params(), indent=4)


    def save_hole_model(self, path):
        torch.save(self.state_dict(), path)

        clss = type(self)
        gen_mod_params = self.get_general_model_params()
        archi_params = self.get_archi_params()
        model_params = {
            "class": str(clss),
            "gen_mod_params": gen_mod_params,
            "archi_params": archi_params,
        }

        xml_str = misc_xml.dict_to_xml("model_infos", model_params)
        path_noext, ext = os.path.splitext(path)
        misc_xml.save_xml_str(path_noext + ".xml", xml_str)
