"""Utility functions for the Wan module."""
import os
import torch
import numpy as np
from PIL import Image
import imageio
import argparse
from torchvision.utils import make_grid

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def tensor_to_rgb(tensor, normalize=True, value_range=(-1, 1)):
    """Convert tensor to rgb values.
    Args:
        tensor: torch.Tensor, will be used as [b, c, h, w] or [b, c, t, h, w].
        normalize: bool, should be true if tensor need to be normalized to (0, 1).
        value_range: tuple, original value range of the tensor, (min, max).
    Returns:
        np.array ([b, c, h, w, 3] or [b, c, t, h, w, 3]): the rgb numpy data.
    """
    tensor = tensor.detach().float().cpu().squeeze()
    ndim = tensor.dim()
    if ndim == 3:
        is_single_image = True
        tensor = tensor.unsqueeze(0)  # [c, h, w] -> [1, c, h, w]
    else:
        is_single_image = False

    if normalize:
        tensor = (tensor - value_range[0]) / (value_range[1] - value_range[0])
    tensor = tensor.clamp(0, 1)

    if tensor.shape[1] == 3:
        # b, c, h, w -> b, h, w, c
        r = tensor[:, 0]
        g = tensor[:, 1]
        b = tensor[:, 2]
        tensor = torch.stack((r, g, b), dim=-1)
    else:
        # single channel
        tensor = tensor.repeat(1, 1, 1, 3)

    images = (tensor.numpy() * 255).astype(np.uint8)
    if is_single_image:
        images = images[0]
    return images

def cache_video(tensor, save_file, fps=16, nrow=1, normalize=True, value_range=(-1, 1)):
    """Cache video from tensor.
    Args:
        tensor: torch.Tensor, will be used as [b, c, t, h, w].
        save_file: str, video file path.
        fps: float, frames per second.
        nrow: int, number of videos in a row.
        normalize: bool, should be true if tensor need to be normalized to (0, 1).
        value_range: tuple, original value range of the tensor, (min, max).
    """
    tensor = tensor.detach().float().cpu()
    assert tensor.ndim == 5, f'Need [bs, c, t, h, w], but got {tensor.shape}'
    os.makedirs(os.path.dirname(os.path.abspath(save_file)), exist_ok=True)

    # extract the inputs
    b, c, t, h, w = tensor.shape
    if b % nrow != 0:
        # append blank videos
        rest = nrow - b % nrow
        tensor = torch.cat([tensor, torch.zeros(rest, c, t, h, w)], dim=0)
        b = tensor.shape[0]

    # prepare grid for each timestep
    images = []
    for i in range(t):
        img = make_grid(tensor[:, :, i], nrow=nrow)  # [3, n*h, n*w]
        img = tensor_to_rgb(img, normalize=normalize, value_range=value_range)
        images.append(img)

    # save to file
    try:
        imageio.mimsave(save_file, images, fps=fps)
    except:
        imageio.mimwrite(save_file, images, fps=fps)

def cache_image(tensor, save_file, nrow=1, normalize=True, value_range=(-1, 1)):
    """Cache image from tensor.
    Args:
        tensor: torch.Tensor, will be used as [b, c, h, w].
        save_file: str, image file path.
        nrow: int, number of images in a row.
        normalize: bool, should be true if tensor need to be normalized to (0, 1).
        value_range: tuple, original value range of the tensor, (min, max).
    """
    tensor = tensor.detach().float().cpu()
    assert tensor.ndim == 4, f'Need [bs, c, h, w], but got {tensor.shape}'
    os.makedirs(os.path.dirname(os.path.abspath(save_file)), exist_ok=True)

    b, c, h, w = tensor.shape
    if b % nrow != 0:
        # append blank images
        rest = nrow - b % nrow
        tensor = torch.cat([tensor, torch.zeros(rest, c, h, w)], dim=0)
        b = tensor.shape[0]

    img = make_grid(tensor, nrow=nrow)  # [3, n*h, n*w]
    img = tensor_to_rgb(img, normalize=normalize, value_range=value_range)
    
    # save to file
    Image.fromarray(img).save(save_file) 