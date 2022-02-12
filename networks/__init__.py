from pathlib import Path
from typing import Union

import torch
from torch import nn


def load_weights(network: nn.Module, model_file: Union[str, Path], *, key: str = None, strict: bool = True, convert: bool = False) -> nn.Module:
    weights = torch.load(model_file)
    if key is not None and key in weights:
        weights = weights[key]
    network.load_state_dict(weights, strict=strict)
    return network


