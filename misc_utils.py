from typing import Iterable

import copy
import random
import torch
import torch.nn as nn

import numpy as np


def seed_all(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class EMA:
    def __init__(self, params: Iterable[nn.Parameter], rate: float = 0.999):
        self.rate = rate
        self.params = list(params)  # reference
        self.ema_params = [
            copy.deepcopy(p).detach().requires_grad_(False) for p in self.params
        ]

    @torch.no_grad()
    def update(self):
        for ema_p, p in zip(self.ema_params, self.params):
            ema_p.mul_(self.rate).add_(p, alpha=1 - self.rate)

    @torch.no_grad()
    def apply(self):
        self.stored_params = [p.clone() for p in self.params]
        for p, ema_p in zip(self.params, self.ema_params):
            p.copy_(ema_p)

    @torch.no_grad()
    def restore(self):
        assert getattr(self, "stored_params") is not None
        for p, stored_p in zip(self.params, self.stored_params):
            p.copy_(stored_p)
        del self.stored_params
