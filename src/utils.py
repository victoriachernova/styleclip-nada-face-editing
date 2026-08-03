import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def set_seed(seed=42):

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) 

def show_grid(images, nrow=4, title=None, save_path=None):

    images = images.detach().cpu()
    images = (images.clamp(-1, 1) + 1) / 2

    grid = vutils.make_grid(images, nrow=nrow, padding=2)

    nrows = (images.shape[0] + nrow - 1) // nrow
    plt.figure(figsize=(nrow * 2.5, nrows * 2.5))
    plt.axis('off')
    if title:
        plt.title(title)
    plt.imshow(grid.permute(1, 2, 0).numpy())
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def save_checkpoint(model, path, extra=None):

    state = {'state_dict': model.state_dict()}
    if extra:
        state.update(extra)
    torch.save(state, path)

def load_checkpoint(model, path, device='cuda'):

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    return checkpoint