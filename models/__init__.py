from .denoising_unet import DenoisingUnetModel
from .DORN_nohints import DORN_nyu_nohints
from .DORN_hints import DORN_nyu_hints, DORN_nyu_hints_Unet, DORN_nyu_histogram_matching
from .DORN_bayesian_opt import DORN_bayesian_opt
# from .DORN_sinkhorn_opt import DORN_sinkhorn_opt
# from .DORN_median_matching import DORN_median_matching
# from .DenseDepth_models import DenseDepth, DenseDepthMedianRescaling, DenseDepthSinkhornOpt
# import future models here


def make_model(model_name, model_params, model_state_dict_fn):
    """
    Make a model from the name, params, and function for getting the state dict (if not None).
    :param model_name: The name of the model's class in the global namespace.
    :param model_params: A dictionary for initializing the model with the appropriate parameters
    :param model_state_dict_fn: If not None, a function that, when called, returns a
                                  state dict for initializing the model.
    :return: An initialized model.
    """
    # model
    model_class = globals()[model_name]
    model = model_class(**model_params)
    if model_state_dict_fn is not None:
        model.load_state_dict(model_state_dict_fn())
    else: # New model - apply initialization
        # m.initialize(model)
        pass # Use default pytorch initialization
    return model


def split_params_weight_bias(model):
    """Split parameters into weight and bias terms,
    in order to apply different regularization."""
    split_params = [{"params": []}, {"params": [], "weight_decay": 0.0}]
    for name, param in model.named_parameters():
        # print(name)
        # print(param)
        if "weight" in name:
            split_params[0]["params"].append(param)
        elif "bias" in name:
            split_params[1]["params"].append(param)
        else:
            raise ValueError("Unknown param type: {}".format(name))
    return split_params
