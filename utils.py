import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np


def show_image(image):
    """
    image has shape C, H, W
    """
    plt.figure()
    image = image.transpose(0, 1).transpose(1, 2).detach().cpu().numpy()
    plt.imshow(image)


def draw_sample_image(x, title):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(make_grid(x.detach().cpu(),
               padding=2, normalize=False), (1, 2, 0)))
