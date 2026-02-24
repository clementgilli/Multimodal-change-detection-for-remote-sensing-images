# General

# Imaging

# Local
from src.model_utils import model_basic


MODEL_NAME_TO_CLASS = {
    "basic": "model_basic"
}


def get_model_from_archi_name(archi, **kwargs):

    archi = archi.lower()
    if archi in MODEL_NAME_TO_CLASS.keys():
        archi = MODEL_NAME_TO_CLASS[archi]

    if archi == "model_basic":
        model = model_basic.Model(**kwargs)

    else:
        raise AssertionError("Unknown architecture " + archi + " [get_model_from_archi_name].")

    print("\nModel arguments:\n", model.disp_all_params())
    print("\nModel layers:\n", model)
    model.to(model.device)

    # if from_pretrained:
    #     model.load_state_dict(torch.load(weights_path, map_location=device))

    print("Total number of parameters:", sum(p.numel() for p in model.parameters()), "\n")

    return model
