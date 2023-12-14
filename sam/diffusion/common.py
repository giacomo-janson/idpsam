import os
import torch


class DiffusionCommon:

    def get_sample_model(self):
        if self.ema is None:
            return self.eps_model
        else:
            return self.ema.ema_model