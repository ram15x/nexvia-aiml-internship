# Nexvia AI/ML internship: two capstone projects

Two projects built for the Nexvia Technologies AI/ML internship, each in its own folder with its own README, tests and setup steps.

| Folder | Project | What it does |
|---|---|---|
| [`AIML-02`](AIML-02) | AI-assisted medical imaging review | Scores chest X-rays and segments the spleen on CT, then shows the result with a heat map to a reviewer who approves or rejects it. Educational only, not for clinical use. |
| [`AIML-03`](AIML-03) | Personalised movie recommendation engine | Finds a few hundred candidate films for a person, ranks them with a deep model, and serves the top 10 through a cached API with weekly retraining. |

## AIML-02: medical imaging review

A reviewer needs a result, a reason for it and a record of what they decided. The system provides all three.

- **Models:** DenseNet-121 (14 chest findings, transfer learning on NIH ChestX-ray14) and a MONAI 3D U-Net (spleen CT).
- **Explanations:** Grad-CAM, SHAP and a guarded Q&A box that sees only the stored scores.
- **Review workflow:** React screen over FastAPI; Approve, Reject and Needs review, each logged with the SHA-256 of the model file.
- **Operations:** Orthanc DICOM connector, DVC for data, Docker (non-root), 106 tests.
- **Results:** chest macro AUROC 0.797 on 999 NIH images (0.805 with mirror averaging); Brier score 0.090 to 0.053 after per-label calibration; spleen Dice 0.887 on 6 held-out cases.

Details, architecture and run steps: [`AIML-02/README.md`](AIML-02/README.md).

## AIML-03: recommendation engine

A heavy model cannot score 34,461 films on every request, so a cheap first stage narrows the field and a smarter second stage orders it.

- **Stage 1:** TensorFlow Recommenders two-tower model with a FAISS HNSW index, plus item-KNN; about 250 candidates per user.
- **Stage 2:** DCN-v2 ranker, with Wide & Deep as the A/B challenger.
- **Serving:** FastAPI, Redis cache, Triton, popularity fallback for new users, "because you liked" explanations.
- **Operations:** Spark feature tables, Airflow weekly retraining behind a quality gate, simulated A/B replay.
- **Results (time split, 6,252 test users):** NDCG@10 0.1272 against 0.0728 for popularity and 0.1091 for the best single retriever (+0.018, 95% CI 0.015 to 0.021). Median latency 10.6 ms uncached.

Details, architecture and run steps: [`AIML-03/README.md`](AIML-03/README.md).

## Live demos

- AIML-02 medical imaging review (lighter Gradio front end): https://ram15x-medical-imaging-ai-review.hf.space/
- AIML-03 movie recommender: https://huggingface.co/spaces/Ram15x/movie-recommender

Both run on Hugging Face's free tier, so the first load can take a few minutes.

## Repository layout

```
AIML-02/   source, tests, docs, screenshots, deliverables (deck)
AIML-03/   source, tests, docs, architecture diagrams, deliverables (deck)
```

Each project has its own dependencies. Install and run from inside the project folder, following that folder's README.

## Presentation decks

- [`AIML-02/deliverables/AIML02_Project1_Final.pptx`](AIML-02/deliverables/AIML02_Project1_Final.pptx)
- [`AIML-03/deliverables/AIML03_Recommendation_Engine.pptx`](AIML-03/deliverables/AIML03_Recommendation_Engine.pptx)

## Notice

AIML-02 is an educational prototype and must not be used for diagnosis or any clinical decision.
