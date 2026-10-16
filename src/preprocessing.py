"""Image preprocessing, filtering, enhancement, and augmentation pipelines for CXR.

Implements CLAHE, Laplacian sharpening, normalization, and deterministic inference transforms.
"""

import json
from io import BytesIO
from pathlib import Path
from typing import Any, List, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def load_class_names(json_path: Union[str, Path]) -> List[str]:
    """Loads target classification class names from a JSON file.

    Args:
        json_path (Union[str, Path]): Path to class_names.json.

    Returns:
        List[str]: List of class label strings.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Class names file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def open_as_pil_grayscale(image_input: Any) -> Image.Image:
    """Opens an image from various input types and converts it to single-channel grayscale.

    Args:
        image_input: Path, bytes, BytesIO, or PIL Image.

    Returns:
        Image.Image: Grayscale PIL image in 'L' mode.
    """
    if isinstance(image_input, Image.Image):
        return image_input.convert("L")
    if isinstance(image_input, (str, Path)):
        return Image.open(image_input).convert("L")
    if hasattr(image_input, "read"):
        # File-like object (e.g. Streamlit UploadedFile or BytesIO)
        content = image_input.read()
        if hasattr(image_input, "seek"):
            image_input.seek(0)
        return Image.open(BytesIO(content)).convert("L")
    if isinstance(image_input, bytes):
        return Image.open(BytesIO(image_input)).convert("L")
    raise ValueError(f"Unsupported image input type: {type(image_input)}")


def preprocess_image(
    image_input: Any,
    enhance: bool = True,
    add_noise: bool = False,
    noise_std: float = 5.0,
    target_size: Tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """Preprocesses a raw chest X-ray image for deep learning model inference.

    Pipeline:
      1. Converts to grayscale.
      2. (Optional) Applies CLAHE (Contrast-Limited Adaptive Histogram Equalization).
      3. (Optional) Applies high-pass spatial sharpening filter.
      4. (Optional, training only) Injects subtle Gaussian noise.
      5. Broadcasts single-channel grayscale to 3 channels.
      6. Resizes to target dimension (224x224).
      7. Normalizes intensities to range [-1.0, 1.0].

    Args:
        image_input: Image source (filepath, PIL image, bytes, or file-like object).
        enhance (bool): Whether to apply CLAHE and spatial sharpening.
        add_noise (bool): Whether to add random Gaussian noise (default False for deterministic inference).
        noise_std (float): Standard deviation of Gaussian noise if enabled.
        target_size (Tuple[int, int]): Spatial dimensions (H, W).

    Returns:
        torch.Tensor: Preprocessed tensor of shape (1, 3, H, W).
    """
    pil_gray = open_as_pil_grayscale(image_input)
    img_np = np.array(pil_gray)

    if enhance:
        # Step A: Contrast-Limited Adaptive Histogram Equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_np = clahe.apply(img_np)

        # Step B: 2D Spatial Sharpening Kernel to emphasize edge textures
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        img_np = cv2.filter2D(img_np, -1, sharpen_kernel)

        # Step C: Optional noise injection (recommended only during training augmentation)
        if add_noise and noise_std > 0:
            noise = np.random.normal(0, noise_std, img_np.shape).astype(np.float32)
            img_np = np.clip(img_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    processed_pil = Image.fromarray(img_np)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    tensor = transform(processed_pil)
    return tensor.unsqueeze(0)  # Shape: (1, 3, 224, 224)
