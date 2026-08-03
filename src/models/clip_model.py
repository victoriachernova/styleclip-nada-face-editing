import torch
import clip

class CLIP:

    def __init__(self, device='cuda'):

        self.device = device

        self.model, self.preprocess = clip.load(
            "ViT-B/32",
            device=device
        )

        self.model.eval()

    def encode_text(self, text):

        tokens = clip.tokenize(text).to(self.device)

        return self.model.encode_text(tokens)

    def encode_image(self, image):

        image = self.prepocess(image).unsqeeze(0).to(self.device)

        return self.model.encode_image(image)