# Calibration, mirror averaging and review thresholds

The classifier's raw scores rank images reasonably well but they are not probabilities, and a flat 0.5 cut-off flags too many images. Two cheap steps address that without retraining the network.

## What is served

1. **Mirror averaging.** Each image is scored as it is and mirrored left to right, and the two logits are averaged. On the 999-image NIH sample this lifts macro AUROC from 0.797 to 0.805. The paired difference is +0.007 with a patient-level 95% interval of +0.003 to +0.012 (500 resamples, `results/calibration/flip_tta_and_head.json`). The cost is one extra forward pass. It can be switched off with `NEXVIA_CLASSIFIER_FLIP_TTA=false`.
2. **Per-label Platt scaling.** A one-feature logistic fit maps the averaged logit to a probability estimate for each label.
3. **Review flag.** For each label the threshold on the calibrated probability is the one that maximises F1 on the fitting data. A flag means "worth a person's look", not "disease present".

Labels with fewer than 20 positives in the fitting sample (Pneumonia 7, Edema 11, Fibrosis 1, Hernia 0) are not calibrated. They keep the raw score and get no probability or flag. The fitted values are in `artifacts/calibration.json`; the API loads them at start-up and runs without them if the file is missing.

## How it was checked

Fitting and scoring on the same 999 images would flatter the result, so every number below comes from patient-grouped 5-fold cross-fitting: each image is scored by calibration fitted on other patients' images. The table averages five different fold assignments (`tools/fit_calibration.py`, `results/calibration/calibration_study.csv`).

| Setting | Macro AUROC | Brier | Micro precision | Micro recall | Micro F1 |
|---|---:|---:|---:|---:|---:|
| Single pass, threshold 0.5 | 0.797 | 0.090 | 0.175 | 0.481 | 0.257 |
| Mirror averaged, threshold 0.5 | 0.805 | 0.090 | 0.178 | 0.472 | 0.258 |
| Mirror averaged, calibrated, per-label thresholds | 0.796 | 0.053 | 0.194 | 0.524 | 0.283 |

The AUROC in the last row is lower only because each fold uses a slightly different monotone mapping when scores are pooled; the shipped single mapping does not change the ranking, so the served AUROC is the 0.805 of the second row. Brier falls from 0.090 to 0.053 because calibrated probabilities are small for rare findings.

## What was tried and not adopted

A per-label logistic head on the frozen 1024-dimensional DenseNet features plus the base logit, cross-fitted the same way, gave macro AUROC 0.788 and 0.778 for two regularisation strengths, below the served 0.797. With 999 images and at most 131 positives per label it overfits, so it was dropped.

## Limits

The calibration is fitted on 999 images from one dataset, so it describes that kind of data and may not hold for another scanner, population or label definition. The thresholds trade precision for recall on this sample's prevalence. The age gap reported in [limits_and_bias.md](limits_and_bias.md) is not addressed by this step.
