import numpy as np
import torch
from transformers import AutoImageProcessor
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

def clip_per_lead(sample, min_val=-4, max_val=4):
    """
    Clips the values of the input sample within a specified range.

    Args:
        sample (np.ndarray or torch.Tensor): Input sample.
        min_val (float): Minimum allowed value.
        max_val (float): Maximum allowed value.

    Returns:
        np.ndarray: Clipped sample.
    """
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32)
    return np.array(torch.clamp(sample, min=min_val, max=max_val))

def normalize_image_24ch(image, xmin, xmax):
    """
    Normalizes an image given the xmin and xmax for each channel.

    Args:
        image (np.ndarray): Input image of shape (C, H, W) with values in any range.     
        np.ndarray: Minimum values of each channel (shape (C, 1, 1)).
        np.ndarray: Maximum values of each channel (shape (C, 1, 1)).

    Returns:
        np.ndarray: Normalized image.
    """
    denom = np.where(xmax > xmin, xmax - xmin, 1)
    normalized_image = (image - xmin) / denom

    return normalized_image.astype(np.float32)

def standardize_image_24ch(image, mean, std):
    """
    Standardizes an image using the given mean and standard deviation for each channel.

    Args:
        image (np.ndarray): Input image of shape (C, H, W).
        mean (np.ndarray or list): Mean values per channel.
        std (np.ndarray or list): Standard deviation values per channel.

    Returns:
        np.ndarray: Standardized image.
    """
    assert image.ndim == 3, f"Expected (C, H, W), got {image.shape}"
    mean = np.array(mean).reshape(-1, 1, 1)
    std = np.array(std).reshape(-1, 1, 1)
    std = np.where(std > 0, std, 1)
    return (image - mean) / std

def spectrogram_channel_expansion(spec_sample):
    """
    Expands a 12-channel spectrogram to 24 channels by separating real and imaginary parts.

    Args:
        spec_sample (np.ndarray): Input spectrogram of shape (12, H, W).

    Returns:
        np.ndarray: Expanded 24-channel spectrogram of shape (24, H, W).
    """
    num_leads, H, W = spec_sample.shape
    assert num_leads == 12, f"Expected 12 leads, but got {num_leads}"
    spec_img_24ch = np.empty((24, H, W))
    spec_img_24ch[0::2, :, :] = np.real(spec_sample)
    spec_img_24ch[1::2, :, :] = np.imag(spec_sample)
    return spec_img_24ch

def compute_x1_x99_spec_dataset(dataset, p1, p2):
    """
    Computes the 1st (x1) and 99th (x99) percentile values per channel across the dataset.

    Args:
        dataset (torch.utils.data.Dataset or torch.utils.data.DataLoader): A dataset of 24-channel spectrograms.
        p1 (float): Percentile for the lower bound (default: 1st percentile).
        p2 (float): Percentile for the upper bound (default: 99th percentile).

    Returns:
        tuple: (x1, x99), each of shape (24, 1, 1).
    """
    p1_values = []
    p2_values = []

    for spec_sample in tqdm(dataset, desc="Computing percentiles", unit="sample"):
        p1_values.append(np.percentile(spec_sample, p1, axis=(1, 2)).astype(np.float32)) 
        p2_values.append(np.percentile(spec_sample, p2, axis=(1, 2)).astype(np.float32))

    p1_values = np.stack(p1_values, axis=0).astype(np.float32)  
    p2_values = np.stack(p2_values, axis=0).astype(np.float32)

    x1 = np.percentile(p1_values, 50, axis=0).astype(np.float32).reshape(24, 1, 1)
    x2 = np.percentile(p2_values, 50, axis=0).astype(np.float32).reshape(24, 1, 1)

    return x1, x2

def compute_mean_std_spec_dataset(dataset, batch_size=512, device="cuda"):
    """
    Computes the mean and standard deviation per channel across the dataset using batched processing.

    Args:
        dataset (torch.utils.data.Dataset or torch.utils.data.DataLoader): A dataset of 24-channel spectrograms.
        batch_size (int, optional): The number of samples to process per batch for efficiency (default: 512).
        device (str, optional): The device to run computations on ("cuda" or "cpu").

    Returns:
        tuple: (mean, std), each of shape (24, 1, 1), computed on the dataset.
    """

    # Initialize accumulators for sum and sum of squares
    sum_values = torch.zeros((24,), dtype=torch.float64, device=device)  # Sum for mean computation
    sum_squares = torch.zeros((24,), dtype=torch.float64, device=device)  # Sum of squares for std computation
    total_pixels = 0  # Total number of elements across all batches

    # Use DataLoader
    if not isinstance(dataset, torch.utils.data.DataLoader):    
        dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch in tqdm(dataset, desc="Computing mean/std", unit="batch"):
        batch = batch.to(device) 
        
        # Compute total number of elements in this batch
        batch_size, num_channels, height, width = batch.shape
        num_elements = batch_size * height * width  # Total elements per channel

        # Sum of all values in the batch
        sum_values += batch.sum(dim=(0, 2, 3))

        # Sum of all squared values in the batch
        sum_squares += (batch ** 2).sum(dim=(0, 2, 3))

        total_pixels += num_elements  # Accumulate total number of elements

    # Compute mean per channel
    mean_values = sum_values / total_pixels

    # Compute standard deviation using std = sqrt(E[X^2] - (E[X])^2)
    std_values = torch.sqrt((sum_squares / total_pixels) - (mean_values ** 2))

    # Reshape for consistency
    mean_values = mean_values.view(24, 1, 1)
    std_values = std_values.view(24, 1, 1)

    return mean_values.float().to("cpu"), std_values.float().to("cpu")

def spectrogram_preprocessing_24ch(spec_sample, mean, std, resize_size=(224, 224), clip_value=3):
    """
    Preprocesses a 24-channel spectrogram sample by normalizing, standardizing, clipping, and resizing.

    Args:
        spec_sample (np.ndarray or torch.Tensor): Input spectrogram sample of shape (24, H, W).
        mean (torch.Tensor or np.ndarray): Mean values per channel (expected shape (24, 1, 1)).
        std (torch.Tensor or np.ndarray): Standard deviation values per channel (expected shape (24, 1, 1)).
        resize_size (tuple): Target size for resizing (H, W), default (224, 224).
        clip_value (float or None): Maximum absolute value for clipping (default 3).

    Returns:
        torch.Tensor: Processed and resized spectrogram of shape (24, resize_size[0], resize_size[1]).
    """

    
    spec_tensor = torch.as_tensor(spec_sample, dtype=torch.float32)

    mean = torch.as_tensor(mean, dtype=torch.float32, device=spec_tensor.device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=torch.float32, device=spec_tensor.device).view(-1, 1, 1)

    
    standardized_sample = (spec_tensor - mean) / std

    # Apply clipping to limit extreme values
    if clip_value is not None:
        standardized_sample = torch.clamp(standardized_sample, -clip_value, clip_value)

    # Resize using bicubic interpolation
    if spec_sample.ndim ==4: # if input is a batch
        resized_img = F.interpolate(standardized_sample, size=resize_size, mode="bicubic", align_corners=False)
    else:
        resized_img = F.interpolate(standardized_sample.unsqueeze(0), size=resize_size, mode="bicubic", align_corners=False).squeeze(0)

    return resized_img