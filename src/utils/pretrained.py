import torch.nn as nn

def remove_last_layers(model, n):
    return nn.Sequential(*(list(model.children())[:-n]))