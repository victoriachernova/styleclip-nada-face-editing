import torch

class MapperTrainerSE:
    def __init__(self,
                 mapper,
                 generator,
                 loss,
                 lr=1e-4,
                 l2_lambda=0.0006,
                 device='cuda'
    ):
        self.mapper = mapper
        self.generator = generator
        self.loss = loss

        self.device = device
        self.l2_lambda = l2_lambda

        self.optimizer = torch.optim.Adam(self.mapper.parameters(), lr=lr)

    def train_step(self,source_text, target_text, style_id):

        self.mapper.train()

        z = torch.randn(1, 
                        self.generator.G.z_dim,
                        device = self.device
                        )
        with torch.no_grad():
             w = self.generator.mapping(z)

        delta_w = self.mapper(w, style_id)

        delta_w = delta_w * torch.clamp(5.0 / (delta_w.norm(dim=-1, keepdim=True) + 1e-8), max=1.0)

        w_edit = w + delta_w

        with torch.no_grad():
            image_before = self.generator.synthesis(w)

        image_after = self.generator.synthesis(w_edit)

        clip_loss = self.loss(image_before, image_after, source_text, target_text)

        l2_loss = delta_w.norm(dim=-1).mean()

        total_loss = clip_loss + self.l2_lambda * l2_loss

        self.optimizer.zero_grad()

        total_loss.backward()

        self.optimizer.step()

        return {
            "total": total_loss.item(),
            "clip": clip_loss.item(),
            "l2": l2_loss.item()
        }