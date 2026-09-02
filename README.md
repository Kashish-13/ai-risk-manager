# AI Risk Manager

AI Risk Manager is a reproducible fraud-detection and transaction-risk scoring project for the Razorpay Buildathon. It will compare a Logistic Regression baseline with a tree-based model, evaluate them on a held-out test set, translate calibrated model probabilities into configurable risk bands, and present evidence-based risk indicators in Streamlit.

## Problem and impact

Payment fraud is a rare-event classification problem: missing fraud can cost merchants money, while false positives interrupt legitimate payments. The project focuses on transparent ranking and review support rather than claiming autonomous fraud proof.

## Dataset

The primary dataset is the public ULB/Worldline Credit Card Fraud benchmark, fetched reproducibly from OpenML dataset 1597. Its records are anonymized and highly imbalanced. See [data/README.md](data/README.md) for its limits and the clearly labelled synthetic development fallback.

The downloaded OpenML representation has 284,807 rows, including 492 fraud labels. Validation found 9,144 exact duplicate rows. The configurable Phase 3 policy is `retain`: these are retained because they have no feature-level label conflicts, and deleting benchmark rows would change the observed class distribution. This decision is logged in each model artifact.

## Planned approach

- Validate input data without silently changing it.
- Create schema-supported, deterministic time and amount features while retaining source fields.
- Use stratified train/validation/test splits and fit all learned preprocessing on training data only.
- Compare a class-weighted Logistic Regression baseline with a tree-based model.
- Report precision, recall, F1, ROC-AUC, PR-AUC, a confusion matrix, and a classification report only after real evaluation.

No model metrics are claimed yet because the model training and held-out evaluation phase has not run.

## Repository layout

`src/` contains reusable data, validation, feature, preprocessing, risk, and prediction contracts. `scripts/` contains reproducible data acquisition and development-only synthetic data generation. Raw data, model files, and generated reports are excluded from Git.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\download_data.py
pytest
```

Training and the Streamlit dashboard are intentionally deferred until their approved phases. Future deployment will use a container-compatible Streamlit command and environment-specific configuration—no credentials committed to the repository.

## Limitations and ethics

The benchmark is anonymized, historical, and not Razorpay data. A high score is a risk indicator, not proof of fraud. Any production use needs monitoring for drift, privacy review, human escalation paths, threshold governance, and fairness assessment.

## Demo placeholder

Screenshots and genuine held-out-test results will be added after model training and dashboard implementation.
