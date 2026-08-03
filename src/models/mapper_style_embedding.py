import torch
import torch.nn as nn


class StyleEmbeddingMapper(nn.Module):
    def __init__(self,
                  latent_dim=512,
                  hidden_dim=1024,
                  style_dim=64,
                  num_styles=3,
                  num_layers=4
    ):
        super().__init__()

        self.style_embedding = nn.Embedding(num_embeddings=num_styles,
                                            embedding_dim=style_dim)

        layers = []

        layers.append(nn.Linear(latent_dim + style_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers-2):

            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, latent_dim))

        self.mapping = nn.Sequential(*layers)

    def forward(self, w, style_id):

        B, L, C = w.shape

        style_id = torch.tensor([style_id], device=w.device)

        style = self.style_embedding(style_id)

        style = style.unsqueeze(1).repeat(B, L, 1)

        x = torch.cat([w, style], dim=-1)

        x = x.view(B * L, -1)

        delta = self.mapping(x)

        delta = delta.view(B, L, C)

        return delta

