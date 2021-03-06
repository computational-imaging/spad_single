import os
import torch


def safe_makedir(path):
    """Makes a directory, or returns if the directory
    already exists.

    Taken from:
    https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory-in-python
    """
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def load_checkpoint(ckpt_file):
    """Loads a checkpoint from a checkpointfile.
    Checkpoint is a dict consisting of:

    model_ckpt
    ----------
    model_name
    model_params
    model_state_dict

    train_ckpt
    ----------
    optim_name
    optim_params
    optim_state_dict

    scheduler_name
    scheduler_params
    last_epoch
    global_it

    ckpt_ckpt
    --------
    run_id
    log_dir
    ckpt_dir

    --
    Can derive model_name and model_state_dict from model
    Can derive scheduler_name from scheduler
       Can derive optim_name and optim_state_dict from scheduler.optimizer

    """
    if torch.cuda.is_available():
        checkpoint = torch.load(ckpt_file,
                                map_location="cuda")
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(ckpt_file,
                                map_location=lambda storage,
                                                    loc: storage)
    model_update = {
        "model_name": checkpoint["model_name"],
        "model_params": checkpoint["model_params"],
        # Because of sacred - don't want to explode the config
        "model_state_dict_fn": lambda: checkpoint["model_state_dict"],
    }
    train_update = {
        "optim_name": checkpoint["optim_name"],
        # Because of sacred - don't want to explode the config
        "optim_state_dict_fn": lambda: checkpoint["optim_state_dict"],
        "scheduler_name": checkpoint["scheduler_name"],
        "scheduler_params": checkpoint["scheduler_params"],
        "last_epoch": checkpoint["last_epoch"],
        "global_it": checkpoint["global_it"],
    }
    ckpt_update = {
        "run_id": checkpoint["run_id"],
        "log_dir": checkpoint["log_dir"],
        "ckpt_dir": checkpoint["ckpt_dir"],
    }
    return model_update, train_update, ckpt_update


def save_checkpoint(model,
                    scheduler,
                    config,
                    state,
                    filename):
    """Save checkpoint using
    model - the current model
    scheduler - the scheduler (and optimizer)
    config - configuration of the model
    state - the current state of the training process
    """
    # Unpack
    model_config = config["model_config"]
    train_config = config["train_config"]
    ckpt_config = config["ckpt_config"]

    checkpoint = {
        "model_name": model.__class__.__name__,
        "model_params": model_config["model_params"],
        "model_state_dict": model.state_dict(),

        "optim_name": scheduler.optimizer.__class__.__name__,
        "optim_params": train_config["optim_params"],
        "optim_state_dict": scheduler.optimizer.state_dict(),
        "scheduler_name": scheduler.__class__.__name__,
        "scheduler_params": train_config["scheduler_params"],
        "last_epoch": state["last_epoch"],
        "global_it": state["global_it"],

        "run_id": ckpt_config["run_id"],
        "log_dir": ckpt_config["log_dir"],
        "ckpt_dir": ckpt_config["ckpt_dir"],
    }
    print("=> Saving checkpoint to: {}".format(filename))
    torch.save(checkpoint, filename)  # save checkpoint
