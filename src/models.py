"""Neural network architectures for CXR lung disease feature extraction and classification.

Contains multi-backbone feature extractors (VGG16, ResNet50, DenseNet121) and
fully connected classification heads (FCNN).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


class MultiBackboneFeatureExtractor(nn.Module):
    """Deep CNN feature extractor supporting VGG16, ResNet50, and DenseNet121 backbones.

    Extracts high-dimensional latent representations from preprocessed chest radiographs.
    Backbone parameters are frozen to prevent catastrophic forgetting when using
    pre-extracted features.

    Attributes:
        backbone (str): Name of the architecture ('vgg16', 'resnet50', or 'densenet121').
        output_dim (int): Dimensionality of the extracted feature vector.
    """

    def __init__(self, backbone: str = "resnet50") -> None:
        super().__init__()
        self.backbone = backbone.lower()

        if self.backbone == "vgg16":
            weights = models.VGG16_Weights.IMAGENET1K_V1
            model = models.vgg16(weights=weights)
            self.features = model.features
            self.pooling = model.avgpool
            self.flatten = nn.Flatten()
            self.fc = nn.Sequential(*list(model.classifier.children())[:-1])
            self.output_dim = 4096

        elif self.backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            model = models.resnet50(weights=weights)
            self.features = nn.Sequential(*list(model.children())[:-1])
            self.flatten = nn.Flatten()
            self.fc = nn.Identity()
            self.output_dim = 2048

        elif self.backbone == "densenet121":
            weights = models.DenseNet121_Weights.IMAGENET1K_V1
            model = models.densenet121(weights=weights)
            self.features = model.features
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()
            self.fc = nn.Identity()
            self.output_dim = 1024

        else:
            raise ValueError(
                f"Unsupported backbone: '{backbone}'. Choose from 'vgg16', 'resnet50', 'densenet121'."
            )

        # Freeze feature extraction weights
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for feature extraction.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor): Extracted feature vector of shape (B, output_dim).
        """
        x = self.features(x)
        if self.backbone in ["vgg16", "densenet121"]:
            x = self.pooling(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class FCNNClassifier(nn.Module):
    """Fully Connected Neural Network (FCNN) classification head for multi-class prediction.

    Consists of 3 dense linear blocks regularized with Batch Normalization,
    LeakyReLU activations, and Dropout to combat overfitting on medical imaging data.

    Args:
        input_dim (int): Dimensionality of the incoming feature vector (e.g. 4096, 2048, 1024).
        num_classes (int): Number of target diagnostic categories (default: 9).
    """

    def __init__(self, input_dim: int, num_classes: int = 9) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through classification layers.

        Args:
            x (torch.Tensor): Feature representations of shape (B, input_dim).

        Returns:
            torch.Tensor: Unnormalized raw logits of shape (B, num_classes).
        """
        return self.classifier(x)


def load_model_weights(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    optional: bool = False,
) -> bool:
    """Safely loads weights into a model checkpoint with fallback handling.

    Args:
        model (nn.Module): Target PyTorch module.
        checkpoint_path (Path): Path to the .pth state dict file.
        device (torch.device): Device to map weights to.
        optional (bool): If True, missing weights will log a warning rather than raising an error.

    Returns:
        bool: True if weights loaded successfully, False otherwise.
    """
    if not checkpoint_path.exists():
        if optional:
            logger.warning(
                "Optional checkpoint not found: '%s'. Keeping default initialized/pretrained weights.",
                checkpoint_path,
            )
            return False
        raise FileNotFoundError(f"Required model checkpoint not found: '{checkpoint_path}'")

    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        # Fallback for older PyTorch versions where weights_only may not be supported or fails
        state_dict = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(state_dict)
    model.eval()
    return True


def load_ensemble_models(
    weights_dir: Path,
    device: torch.device,
    backbone_names: Optional[List[str]] = None,
    num_classes: int = 9,
) -> Tuple[Dict[str, Tuple[MultiBackboneFeatureExtractor, FCNNClassifier]], List[str]]:
    """Loads all feature extractors and classifiers for the ensemble pipeline.

    Args:
        weights_dir (Path): Directory containing `.pth` weight checkpoints.
        device (torch.device): Target device (CPU or CUDA).
        backbone_names (Optional[List[str]]): List of backbones to load.
        num_classes (int): Number of target classification categories.

    Returns:
        Tuple: (models_dict, warnings_list)
            models_dict maps backbone name to (extractor, classifier) on device.
            warnings_list contains any non-fatal notices (e.g. missing optional weights).
    """
    if backbone_names is None:
        backbone_names = ["vgg16", "resnet50", "densenet121"]

    models_dict = {}
    warnings_list = []

    for name in backbone_names:
        extractor = MultiBackboneFeatureExtractor(backbone=name)
        extractor_path = weights_dir / f"extractor_{name}.pth"

        # Check if fine-tuned extractor weights exist; if not, use pretrained ImageNet
        loaded_ext = load_model_weights(extractor, extractor_path, device, optional=True)
        if not loaded_ext:
            warnings_list.append(
                f"Notice: '{extractor_path.name}' was not found. Using default pretrained ImageNet feature extractor for {name.upper()}."
            )

        classifier = FCNNClassifier(input_dim=extractor.output_dim, num_classes=num_classes)
        classifier_path = weights_dir / f"{name}_classifier.pth"
        load_model_weights(classifier, classifier_path, device, optional=False)

        extractor.to(device)
        classifier.to(device)
        extractor.eval()
        classifier.eval()

        models_dict[name] = (extractor, classifier)

    return models_dict, warnings_list
