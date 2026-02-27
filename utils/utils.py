import torch
from torch import nn

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
    
def get_optimizer(optimizer_name: str, model_parameters, lr=0.001, **kwargs):
    params = list(model_parameters)
    if not params:
        raise ValueError("Model parameters are empty.")

    name = optimizer_name.lower()
    
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, **kwargs)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, **kwargs)
    elif name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, **kwargs)
    else:
        return torch.optim.Adam(params, lr=lr, **kwargs)
    
def get_loss_function(loss_name: str):
    losses = {
        "cross_entropy": nn.CrossEntropyLoss(),
        "mse": nn.MSELoss(),
        "L1": nn.L1Loss()
    }

    return losses.get(loss_name.lower(), nn.CrossEntropyLoss())

def get_scheduler(optimizer, scheduler_name, **kwargs):
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    elif scheduler_name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    return None