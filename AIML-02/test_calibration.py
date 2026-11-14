import json
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torch import nn

from api import app as api_module
from api.calibration import Calibration

SPEC = {
    "method": "test",
    "fit_on": "test",
    "labels": {
        "Atelectasis": {"calibrated": True, "a": 1.0, "b": 0.0, "threshold": 0.4, "positives": 80},
        "Hernia": {"calibrated": False, "positives": 0},
    },
}


def test_calibrated_label_gets_probability_and_flag():
    cal = Calibration(SPEC)
    high = cal.apply("Atelectasis", 1.0)
    low = cal.apply("Atelectasis", -1.0)
    assert abs(high["calibrated_probability"] - 0.731059) < 1e-6
    assert high["review_flag"] is True
    assert low["review_flag"] is False


def test_label_without_calibration_has_no_probability_or_flag():
    cal = Calibration(SPEC)
    assert cal.apply("Hernia", 3.0) == {"calibrated_probability": None, "review_flag": None}
    assert cal.apply("Unknown_label", 3.0) == {"calibrated_probability": None, "review_flag": None}


def test_missing_or_broken_file_falls_back_to_no_calibration(tmp_path):
    assert not Calibration.load(tmp_path / "nope.json").available
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert not Calibration.load(bad).available


def test_shipped_calibration_covers_all_labels_and_skips_rare_ones():
    spec = json.loads(api_module.CALIBRATION_PATH.read_text(encoding="utf-8"))
    assert set(spec["labels"]) == set(api_module.LABELS)
    for entry in spec["labels"].values():
        if entry["positives"] < spec["min_positives"]:
            assert entry["calibrated"] is False
        else:
            assert entry["calibrated"] is True and 0 < entry["threshold"] < 1


class LeftHeavyModel(nn.Module):
    """Gives a different logit to the original and the flipped image."""

    def forward(self, images):
        left = images[:, 0, :, :images.shape[3] // 2].mean()
        right = images[:, 0, :, images.shape[3] // 2:].mean()
        value = 4.0 * (left - right)
        return torch.full((images.shape[0], 14), float(value))


def _image():
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    pixels[:, :16] = 255
    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return Image.open(BytesIO(buffer.getvalue()))


def test_flip_average_cancels_a_left_right_asymmetric_logit(monkeypatch):
    monkeypatch.setattr(api_module, "_load_model", lambda: LeftHeavyModel())
    monkeypatch.setattr(api_module, "_model_metadata", {"model_name": "t"})
    monkeypatch.setattr(api_module, "CLASSIFIER_FLIP_TTA", True)
    averaged = api_module._score_image(_image())
    monkeypatch.setattr(api_module, "CLASSIFIER_FLIP_TTA", False)
    single = api_module._score_image(_image())
    assert all(abs(item["score"] - 0.5) < 1e-6 for item in averaged["findings"])
    assert all(abs(item["score"] - 0.5) > 0.1 for item in single["findings"])
    assert averaged["flip_averaged"] is True and single["flip_averaged"] is False


def test_score_response_carries_calibration_fields(monkeypatch):
    monkeypatch.setattr(api_module, "_load_model", lambda: LeftHeavyModel())
    monkeypatch.setattr(api_module, "_model_metadata", {"model_name": "t"})
    monkeypatch.setattr(api_module, "CALIBRATION", Calibration(SPEC))
    body = api_module._score_image(_image())
    by_label = {item["label"]: item for item in body["findings"]}
    assert by_label["Atelectasis"]["calibrated_probability"] is not None
    assert by_label["Hernia"]["review_flag"] is None
    assert body["calibration"]["available"] is True
