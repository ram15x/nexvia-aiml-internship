"""Save logits (original and flipped) for a manifest of 224x224 images, for calibration work.

Reads the 224x224 RGB uint8 array made from the NIH sample with the same resize as the
served path, runs the checkpoint on each image and its horizontal flip, and writes
logits, flipped logits and the 1024-d pooled features.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from api.modeling import image_to_tensor  # noqa: F401  (same normalisation constants)
from api.modeling import IMAGENET_MEAN, IMAGENET_STD
from train_classifier import make_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-npz", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    x_all = np.load(args.images_npz)["X"]
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = make_model(pretrained=False, architecture=ck["architecture"])
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    logits, flipped, feats = [], [], []
    with torch.inference_mode():
        for i in range(0, len(x_all), 32):
            x = ((x_all[i:i + 32].astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(0, 3, 1, 2)
            x = torch.from_numpy(np.ascontiguousarray(x))
            f = torch.relu(model.features(x)).mean(dim=(2, 3))
            logits.append(model.classifier(f).numpy())
            feats.append(f.numpy())
            flipped.append(model(torch.flip(x, dims=[3])).numpy())
    np.savez(args.output, logits=np.concatenate(logits), logits_flip=np.concatenate(flipped),
             feats=np.concatenate(feats))


if __name__ == "__main__":
    main()
