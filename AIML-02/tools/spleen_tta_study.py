"""Spleen inference options chosen on the validation cases, then reported once on the test cases.

Probabilities are computed once per case for the direct image and for flips along each axis
and cached. Options: which flips to average, the foreground threshold, and whether to fill
holes after keeping the largest connected component.

Run: python tools/spleen_tta_study.py --root <folder with imagesTr/labelsTr> --split results/spleen/case_split.json \
     --checkpoint artifacts/best_spleen_unet_20261109.pt --output results/spleen/tta_study.json --cache <dir>
"""
from __future__ import annotations
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np, torch
from scipy import ndimage
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_segmentation import build_transform, make_model, seed_everything

FLIPS = {"none": (), "x": (2,), "y": (3,), "z": (4,)}


def dice(pred, ref):
    d = int(pred.sum()) + int(ref.sum())
    return 1.0 if d == 0 else 2.0 * int(np.logical_and(pred, ref).sum()) / d


def largest_component(mask):
    labels, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    return labels == (1 + int(np.argmax(sizes)))


def combine(probs, names):
    return np.mean([probs[n].astype(np.float32) for n in names], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    a = ap.parse_args()
    a.cache.mkdir(parents=True, exist_ok=True)
    seed_everything(42)
    split = json.loads(a.split.read_text())
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    patch = tuple(ck["patch_size"])
    model = make_model(ck["architecture"]); model.load_state_dict(ck["state_dict"], strict=True); model.eval()

    data = {}
    for part in ("validation", "test"):
        cases = [{"image": str(a.root / "imagesTr" / c), "label": str(a.root / "labelsTr" / c), "case_id": c} for c in split[part]]
        loader = DataLoader(Dataset(cases, transform=build_transform(False, patch)), batch_size=1)
        data[part] = []
        with torch.inference_mode():
            for b in loader:
                cid = b["case_id"][0]
                f = a.cache / f"{cid}.npz"
                ref = (b["label"][0, 0] == 1).numpy()
                if f.exists():
                    probs = {k: v for k, v in np.load(f).items()}
                else:
                    probs = {}
                    for name, dims in FLIPS.items():
                        img = torch.flip(b["image"], dims=dims) if dims else b["image"]
                        p = torch.softmax(sliding_window_inference(img, patch, 1, model, overlap=0.25), dim=1)[:, 1]
                        if dims:
                            p = torch.flip(p, dims=tuple(d - 1 for d in dims))
                        probs[name] = p[0].numpy().astype(np.float16)
                    np.savez(f, **probs)
                data[part].append({"case": cid, "ref": ref, "probs": {k: v.astype(np.float16) for k, v in probs.items()}})
                print(part, cid, flush=True)

    configs = []
    for tta_name, names in {"none": ["none"], "flip_x": ["none", "x"], "flip_xyz": ["none", "x", "y", "z"]}.items():
        for thr in (0.4, 0.5, 0.6):
            for post in ("largest", "largest+fill"):
                configs.append((tta_name, names, thr, post))

    def run(cfg, part):
        _, names, thr, post = cfg
        scores = []
        for item in data[part]:
            m = combine(item["probs"], names) >= thr
            m = largest_component(m)
            if post == "largest+fill":
                m = ndimage.binary_fill_holes(m)
            scores.append(dice(m, item["ref"]))
        return scores

    rows = []
    for cfg in configs:
        v, t = run(cfg, "validation"), run(cfg, "test")
        rows.append({"tta": cfg[0], "threshold": cfg[2], "post": cfg[3],
                     "val_mean": float(np.mean(v)), "test_mean": float(np.mean(t)), "test_cases": t})
    rows.sort(key=lambda r: -r["val_mean"])
    served = next(r for r in rows if r["tta"] == "none" and r["threshold"] == 0.5 and r["post"] == "largest")
    out = {"selected_on": "validation (6 cases)", "chosen": rows[0], "served_before": served, "all": rows}
    a.output.write_text(json.dumps(out, indent=2))
    for r in rows:
        print(f'{r["tta"]:9s} thr {r["threshold"]} {r["post"]:13s} val {r["val_mean"]:.4f} test {r["test_mean"]:.4f}')


if __name__ == "__main__":
    main()
