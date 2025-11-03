import numpy as np
import torch
from transformers import AutoImageProcessor
import torch.nn.functional as F
from tqdm import tqdm

def denormalize_image_24ch(normalized_image, original_min, original_max):
    """
    Denormalizes a 24-channel image back to its original range.

    Args:
        normalized_image (np.ndarray): Normalized image of shape (24, H, W).
        original_min (np.ndarray): Minimum values per channel (shape (24, 1, 1)).
        original_max (np.ndarray): Maximum values per channel (shape (24, 1, 1)).

    Returns:
        np.ndarray: Denormalized image with values in its original range (24, H, W).
    """
    assert normalized_image.ndim == 3, f"Expected (C, H, W), got {normalized_image.shape}"
    original_min = np.array(original_min).reshape(24, 1, 1)
    original_max = np.array(original_max).reshape(24, 1, 1)
    scale = np.where(original_max > original_min, original_max - original_min, 1)
    return normalized_image * scale + original_min

def destandardize_image_24ch(standardized_image, mean, std):
    """
    Reverses standardization for a 24-channel image.

    Args:
        standardized_image (np.ndarray): Standardized image of shape (24, H, W).
        mean (np.ndarray | list | torch.Tensor): Mean values for each channel (shape (24,) or scalar).
        std (np.ndarray | list | torch.Tensor): Standard deviation values for each channel (shape (24,) or scalar).

    Returns:
        np.ndarray: De-standardized image with values in its original range (24, H, W).
    """
    assert standardized_image.ndim == 3, f"Expected (C, H, W), got {standardized_image.shape}"
    mean = np.array(mean).reshape(24, 1, 1)
    std = np.array(std).reshape(24, 1, 1)
    std = np.where(std > 0, std, 1)
    return (standardized_image * std) + mean


def spectrogram_postprocessing_24ch(resized_img, mean, std, original_size=(129, 38)):
    """
    Post-processes a spectrogram image by resizing it back, reversing standardization, and restoring original values.

    Args:
        resized_img (torch.Tensor): Resized standardized image of shape (24, H, W).
        mean (np.ndarray | list | torch.Tensor): Mean values used for standardization (shape (24,) or scalar).
        std (np.ndarray | list | torch.Tensor): Standard deviation values used for standardization (shape (24,) or scalar).
        original_size (tuple[int, int], optional): Target size to resize back to (height, width). Default is (129, 38).

    Returns:
        np.ndarray: Fully post-processed image with original values and dimensions (24, original_size[0], original_size[1]).
    """
    if resized_img.ndim == 3:
        resized_img = resized_img.unsqueeze(0)

    # Resize back to original spectrogram size
    restored_size = F.interpolate(resized_img, size=original_size, mode="bicubic", align_corners=False).squeeze(0)

    # Convert to NumPy if it's a PyTorch tensor
    # restored_array = restored_size.cpu().detach().numpy()
    # Reverse Standardization: x_original = x_processed * std + mean
    # original_spectrogram = (restored_array * std) + mean

    if isinstance(mean, np.ndarray) or isinstance(mean, list):
        mean = torch.tensor(mean, dtype=restored_size.dtype, device=restored_size.device)
    if isinstance(std, np.ndarray) or isinstance(std, list):
        std = torch.tensor(std, dtype=restored_size.dtype, device=restored_size.device)
    # reshape mean/std to broadcast
    while mean.ndim < restored_size.ndim:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    # Reverse Standardization
    original_spectrogram = restored_size * std + mean


    return original_spectrogram  

# def spectrogram_postprocessing_24ch(resized_img, mean, std, original_size=(129, 38)):
#     """
#     Avoid resize; just reverse standardization.
#     """
#     if resized_img.ndim == 3:
#         resized_img = resized_img.unsqueeze(0)

#     # Skip interpolate if original_size == resized_img.shape[-2:]
#     if resized_img.shape[-2:] != original_size:
#         restored_size = F.interpolate(resized_img, size=original_size, mode="bilinear", align_corners=False)
#     else:
#         restored_size = resized_img

#     if isinstance(mean, (np.ndarray, list)):
#         mean = torch.tensor(mean, dtype=restored_size.dtype, device=restored_size.device)
#     if isinstance(std, (np.ndarray, list)):
#         std = torch.tensor(std, dtype=restored_size.dtype, device=restored_size.device)

#     while mean.ndim < restored_size.ndim:
#         mean = mean.unsqueeze(0)
#         std = std.unsqueeze(0)

#     return restored_size * std + mean  # No .squeeze()