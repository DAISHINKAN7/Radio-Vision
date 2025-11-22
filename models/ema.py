"""
Exponential Moving Average (EMA) for model weights
"""

import torch
import torch.nn as nn
from copy import deepcopy


class ModelEMA:
    """Exponential Moving Average of model parameters"""
    def __init__(self, model, decay=0.9999, device=None):
        self.ema = deepcopy(model).eval()
        self.decay = decay
        self.device = device

        if self.device is not None:
            self.ema.to(device)

        for param in self.ema.parameters():
            param.requires_grad_(False)

    def update(self, model):
        """Update EMA parameters"""
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema.parameters(), model.parameters()):
                if self.device is not None:
                    model_param = model_param.to(self.device)
                ema_param.copy_(ema_param * self.decay + model_param * (1 - self.decay))

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state_dict):
        self.ema.load_state_dict(state_dict)

    @property
    def module(self):
        return self.ema
