import torch
import torch.nn.functional as F
import clip

class DirectCLIPloss(torch.nn.Module):

    def __init__(self, stylegan_size=1024, device='cuda'):
        super().__init__()

        self.device = device

        self.model, self.preprocess = clip.load("ViT-B/32", device=device)

        self.upsample = torch.nn.Upsample(scale_factor=7)

        self.avgpool = torch.nn.AvgPool2d(kernel_size=stylegan_size//32)

    def encode_image(self, image):

        image = self.avgpool(self.upsample(image))
        image = (image + 1) / 2

        feat = self.model.encode_image(image)

        feat = feat / feat.norm(dim=1, keepdim=True)

        return feat 

    def encode_text(self, text):

        tokens = clip.tokenize(text).to(self.device)

        feat = self.model.encode_text(tokens)
        
        feat = feat / feat.norm(dim=1, keepdim=True)

        return feat 

    def forward(self, image_before, image_after, source_text, target_text):

        image_before_f = self.encode_image(image_before)
        image_after_f = self.encode_image(image_after)

        text_before_f = self.encode_text(source_text)
        text_after_f = self.encode_text(target_text)

        delta_i = image_after_f - image_before_f
        delta_t = text_after_f - text_before_f

        delta_i = delta_i / (delta_i.norm(dim=1, keepdim=True) + 1e-8)
        delta_t = delta_t / (delta_t.norm(dim=1, keepdim=True) + 1e-8)

        sim = (delta_i * delta_t).sum(dim=-1)

        loss = (1 - sim).mean()

        return loss