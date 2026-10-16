"""Explainable Artificial Intelligence (XAI) for Medical CXR Interpretability.

Implements fast batched Occlusion Sensitivity Mapping to identify the thoracic regions
that most heavily influence the deep learning models' diagnostic classifications.
"""

from typing import Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


@torch.no_grad()
def compute_occlusion_map_fast(
    image_tensor: torch.Tensor,
    extractor: nn.Module,
    classifier: nn.Module,
    target_class_idx: int,
    device: torch.device,
    patch_size: int = 30,
    stride: int = 30,
    downsample: int = 112,
    batch_size: int = 16,
) -> np.ndarray:
    """Computes an accelerated Occlusion Sensitivity Heatmap for a given model.

    Mathematical Formulation:
        Let I be the original chest radiograph, and let I_{x,y} be the radiograph with
        a rectangular patch centered at (x, y) occluded (zeroed out).
        The sensitivity drop at spatial coordinates (x, y) is defined as:
            Delta P_c(x, y) = max(0, P_c(I) - P_c(I_{x,y}))

    Optimizations for clinical responsiveness:
      - Spatial downsampling to an intermediate resolution (e.g., 112x112).
      - Batched parallel forward passes through the network instead of sequential loops.
      - Bilinear upsampling of the resulting sensitivity grid back to input size.

    Args:
        image_tensor (torch.Tensor): Preprocessed input tensor of shape (1, 3, H, W).
        extractor (nn.Module): CNN feature extractor backbone.
        classifier (nn.Module): FCNN classification head.
        target_class_idx (int): Integer index of target disease category.
        device (torch.device): Compute device (CPU or CUDA).
        patch_size (int): Spatial dimensions of the occluding square patch in pixels.
        stride (int): Stride step size between consecutive occlusion patches.
        downsample (int): Intermediate spatial resolution for occlusion testing.
        batch_size (int): Number of occluded variations evaluated per forward batch.

    Returns:
        np.ndarray: 2D normalized sensitivity heatmap of shape (H, W) in range [0.0, 1.0].
    """
    extractor.eval()
    classifier.eval()
    image_tensor = image_tensor.to(device)

    _, _, orig_h, orig_w = image_tensor.shape

    # Baseline probability on non-occluded image
    base_logits = classifier(extractor(image_tensor))
    base_probs = torch.softmax(base_logits, dim=1).squeeze(0)
    base_conf = base_probs[target_class_idx].item()

    # Downsample input tensor for faster occlusion grid generation
    img_small = torch.nn.functional.interpolate(
        image_tensor, size=(downsample, downsample), mode="bilinear", align_corners=False
    )
    _, _, h, w = img_small.shape
    heatmap_small = np.zeros((h, w), dtype=np.float32)

    # Precalculate top-left patch anchor coordinates
    positions = []
    for y in range(0, max(1, h - patch_size + 1), stride):
        for x in range(0, max(1, w - patch_size + 1), stride):
            positions.append((x, y))

    if not positions:
        return np.zeros((orig_h, orig_w), dtype=np.float32)

    def process_batch(batch_list):
        batch_t = torch.cat(batch_list, dim=0).to(device)
        with torch.no_grad():
            logits = classifier(extractor(batch_t))
            probs = torch.softmax(logits, dim=1)
        return probs[:, target_class_idx].cpu().numpy()

    occluded_batch = []
    all_drops = []

    for i, (x, y) in enumerate(positions):
        occluded = img_small.clone()
        # Zero out patch region
        occluded[:, :, y : y + patch_size, x : x + patch_size] = 0.0
        occluded_batch.append(occluded)

        if len(occluded_batch) == batch_size or i == len(positions) - 1:
            batch_confs = process_batch(occluded_batch)
            all_drops.extend([base_conf - val for val in batch_confs])
            occluded_batch = []

    # Map probability drops onto spatial sensitivity map
    for (x, y), drop in zip(positions, all_drops):
        heatmap_small[y : y + patch_size, x : x + patch_size] = np.maximum(
            heatmap_small[y : y + patch_size, x : x + patch_size], drop
        )

    # ReLU rectifying and min-max normalization
    heatmap_small = np.maximum(heatmap_small, 0.0)
    max_val = heatmap_small.max()
    if max_val > 1e-8:
        heatmap_small /= max_val

    # Upsample sensitivity map back to original input dimensions
    heatmap_resized = cv2.resize(heatmap_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return heatmap_resized


def apply_heatmap_overlay(
    original_image: Union[np.ndarray, Image.Image],
    heatmap: np.ndarray,
    alpha: float = 0.6,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlays a 2D sensitivity heatmap onto an original image using pseudo-coloring.

    Args:
        original_image: Background image as RGB numpy array or PIL Image.
        heatmap (np.ndarray): 2D float array in range [0, 1].
        alpha (float): Blend weight for the original image (1 - alpha for heatmap).
        colormap (int): OpenCV colormap enum (e.g. cv2.COLORMAP_JET).

    Returns:
        np.ndarray: Blended RGB image uint8 array of shape (H, W, 3).
    """
    if isinstance(original_image, Image.Image):
        img_np = np.array(original_image.convert("RGB"))
    else:
        img_np = original_image.copy()

    # Resize image to match heatmap dimensions if necessary
    h, w = heatmap.shape[:2]
    if img_np.shape[:2] != (h, w):
        img_np = cv2.resize(img_np, (w, h))

    # Convert heatmap to 8-bit colormap
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    # Convert BGR (OpenCV) to RGB
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend original and colored heatmap
    overlay = cv2.addWeighted(img_np, alpha, heatmap_colored, 1.0 - alpha, 0)
    return overlay
