import torch
import torch.nn as nn

class Mapper(nn.Module):
    def __init__(self, latent_dim=512, hidden_dim=1024, num_layers=4):
        super().__init__()

        layers = [
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU()
        ]

        for _ in range(num_layers-2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, latent_dim))

        self.mapping = nn.Sequential(*layers)

    def forward(self, w):
        
        B, L, C = w.shape

        w = w.view(B*L, C)

        delta = self.mapping(w)

        delta = delta.view(B, L, C)

        return delta