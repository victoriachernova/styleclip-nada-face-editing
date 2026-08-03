import torch
import sys
import os


sys.path.append('/content/stylegan2-ada-pytorch')

import legacy

class Generator:

    def __init__(self, model_path, device='cuda'):

        self.device = device

        with open(model_path, 'rb') as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(device)

        self.G.eval()

    def mapping(self, z, c=None, truncation_psi=1.0):
        return self.G.mapping(z, c, truncation_psi=truncation_psi)
    
    def synthesis(self, w):
        return self.G.synthesis(w)

    def forward(self, z):
        img = self.G(z, None)
        return img
    
    def __call__(self, z):
        return self.forward(z)
    

    