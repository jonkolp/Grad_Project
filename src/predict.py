"""Command-line inference script and standalone programmatic API for CXR lung disease prediction.

Usage (CLI):
    python src/predict.py --image Images/1.png --weights_dir GUI --output_heatmap heatmap_output.png
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from src.explainability import apply_heatmap_overlay, compute_occlusion_map_fast
from src.models import FCNNClassifier, MultiBackboneFeatureExtractor, load_ensemble_models
from src.preprocessing import load_class_names, preprocess_image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def predict_single_model(
    image_tensor: torch.Tensor,
    extractor: MultiBackboneFeatureExtractor,
    classifier: FCNNClassifier,
    device: torch.device,
) -> np.ndarray:
    """Computes softmax class probability distribution for a single backbone pipeline.

    Args:
        image_tensor (torch.Tensor): Preprocessed image tensor (1, 3, H, W).
        extractor: Feature extractor module.
        classifier: Classification head module.
        device: Compute device.

    Returns:
        np.ndarray: Array of probabilities across classes.
    """
    extractor.eval()
    classifier.eval()
    image_tensor = image_tensor.to(device)
    features = extractor(image_tensor)
    logits = classifier(features)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    return probs.cpu().numpy()


def predict_ensemble(
    image_input: Union[str, Path, bytes],
    models_dict: Dict[str, Tuple[MultiBackboneFeatureExtractor, FCNNClassifier]],
    class_names: List[str],
    weights: Optional[Dict[str, float]] = None,
    device: Optional[torch.device] = None,
    enhance: bool = True,
) -> Dict:
    """Performs full ensemble inference on an input chest radiograph.

    Args:
        image_input: Image path, bytes, or PIL Image.
        models_dict: Loaded dictionary of model backbones and classifiers.
        class_names: List of class name strings.
        weights: Optional dictionary of weights for each backbone (e.g. {'vgg16': 0.33, ...}).
        device: Compute device.
        enhance: Whether to apply CLAHE and spatial sharpening.

    Returns:
        Dict: Full diagnostic report containing:
            - 'predicted_class': Str top class label.
            - 'confidence': Float top class probability.
            - 'ensemble_probabilities': Dict mapping class names to probabilities.
            - 'individual_models': Dict with per-model predictions and distributions.
            - 'normalized_weights': Dict of actual weights used.
            - 'image_tensor': Preprocessed PyTorch tensor.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if weights is None:
        weights = {"vgg16": 0.33, "resnet50": 0.33, "densenet121": 0.34}

    total_w = sum(weights.values())
    norm_weights = {k: v / total_w for k, v in weights.items()} if total_w > 0 else {k: 1.0 / len(weights) for k in weights}

    # Deterministic preprocessing for inference (no random noise)
    image_tensor = preprocess_image(image_input, enhance=enhance, add_noise=False)

    model_outputs = {}
    weighted_probs = []

    for name, (extractor, classifier) in models_dict.items():
        probs = predict_single_model(image_tensor, extractor, classifier, device)
        model_outputs[name] = {
            "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
            "top_class": class_names[int(np.argmax(probs))],
            "top_confidence": float(np.max(probs)),
            "raw_probs": probs,
        }
        weighted_probs.append(probs * norm_weights.get(name, 1.0 / len(models_dict)))

    # Ensemble probability summation
    avg_probs = np.sum(np.stack(weighted_probs), axis=0)
    top_idx = int(np.argmax(avg_probs))

    return {
        "predicted_class": class_names[top_idx],
        "confidence": float(avg_probs[top_idx]),
        "ensemble_probabilities": {class_names[i]: float(avg_probs[i]) for i in range(len(class_names))},
        "individual_models": model_outputs,
        "normalized_weights": norm_weights,
        "image_tensor": image_tensor,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Inference for CXR Lung Disease Multi-Backbone Ensemble Classifier."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to chest X-ray image (JPG/PNG).")
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="GUI",
        help="Path to directory containing .pth weights and class_names.json.",
    )
    parser.add_argument(
        "--class_names",
        type=str,
        default=None,
        help="Path to class_names.json (defaults to <weights_dir>/class_names.json).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cpu', 'cuda'). Defaults to auto-detection.",
    )
    parser.add_argument(
        "--output_heatmap",
        type=str,
        default=None,
        help="Optional filepath to save occlusion sensitivity heatmap overlay image.",
    )
    parser.add_argument(
        "--w_vgg", type=float, default=0.33, help="Ensemble weight for VGG-16."
    )
    parser.add_argument(
        "--w_resnet", type=float, default=0.33, help="Ensemble weight for ResNet-50."
    )
    parser.add_argument(
        "--w_densenet", type=float, default=0.34, help="Ensemble weight for DenseNet-121."
    )

    args = parser.parse_args()

    weights_dir = Path(args.weights_dir).resolve()
    class_names_path = (
        Path(args.class_names).resolve()
        if args.class_names
        else weights_dir / "class_names.json"
    )

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load metadata and models
    class_names = load_class_names(class_names_path)
    models_dict, warnings = load_ensemble_models(weights_dir, device, num_classes=len(class_names))
    for w in warnings:
        logger.warning(w)

    weights = {"vgg16": args.w_vgg, "resnet50": args.w_resnet, "densenet121": args.w_densenet}

    logger.info("Running inference on image: %s", args.image)
    result = predict_ensemble(args.image, models_dict, class_names, weights=weights, device=device)

    print("\n" + "=" * 60)
    print(" 🩻 ENSEMBLE DIAGNOSTIC PREDICTION RESULT")
    print("=" * 60)
    print(f" Predicted Class : {result['predicted_class']}")
    print(f" Confidence      : {result['confidence']:.2%}")
    print("\n Individual Backbone Predictions:")
    for name, data in result["individual_models"].items():
        print(f"  • {name.upper():<12}: {data['top_class']} ({data['top_confidence']:.2%})")

    print("\n Top 3 Ensemble Probabilities:")
    sorted_probs = sorted(
        result["ensemble_probabilities"].items(), key=lambda item: item[1], reverse=True
    )
    for cls_name, prob in sorted_probs[:3]:
        print(f"  - {cls_name:<40}: {prob:.2%}")
    print("=" * 60)

    # Optional Occlusion Heatmap Generation
    if args.output_heatmap:
        logger.info("Generating Occlusion Sensitivity Map for: %s", result["predicted_class"])
        target_idx = class_names.index(result["predicted_class"])
        # Compute heatmap using best performing model (ResNet-50)
        ext, clf = models_dict["resnet50"]
        heatmap = compute_occlusion_map_fast(
            result["image_tensor"], ext, clf, target_idx, device, patch_size=30, stride=30
        )
        orig_img = cv2.imread(args.image)
        overlay = apply_heatmap_overlay(orig_img, heatmap)
        cv2.imwrite(args.output_heatmap, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        logger.info("Saved heatmap overlay to: %s", args.output_heatmap)


if __name__ == "__main__":
    main()
