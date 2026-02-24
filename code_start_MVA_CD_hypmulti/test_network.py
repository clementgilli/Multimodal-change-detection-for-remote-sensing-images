# General
import os

# Imaging

# Local
from src.train_test_utils import train_test_utils


if __name__ == "__main__":

    # ============================ #
    # ----- Saved model path ----- #
    # ============================ #
    curr_path = os.path.dirname(os.path.realpath(__file__))  # Current path
    save_folder = "sample"
    save_dir = os.path.join(curr_path, save_folder)

    # name_model2use = "trained_model.pth"
    name_model2use = "trained_model_best_val.pth"
    # name_model2use = "trained_model_best_train.pth"
    # name_model2use = "trained_model_end.pth"


    # ====================== #
    # ----- Data paths ----- #
    # ====================== #
    data_path = curr_path
    # test_data = "test"
    test_data = "test/numpy_array_RED_val_inputs.npy"
    # train_inp_data = "test/numpy_array_RED_val_inputs.npy"
    # train_tar_data = "test/numpy_array_RED_val_targets.npy"


    # ================= #
    # ----- Model ----- #
    # ================= #
    model = train_test_utils.load_model_path(os.path.join(save_dir, name_model2use))
    print("model", model)


    # ===================== #
    # ----- Inference ----- #
    # ===================== #
    res = model.infer_datas_from_test_dir(os.path.join(data_path, test_data))
