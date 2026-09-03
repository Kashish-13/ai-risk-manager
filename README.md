# AI Risk Manager

A production-style fraud-risk decision-support dashboard created for the Razorpay Buildathon. It analyzes credit-card transaction signals, estimates fraud probability, converts it to a transparent 0–100 risk score, and presents genuine held-out evaluation results in a premium Streamlit interface.

> This is a portfolio and decision-support prototype—not a production banking authorization engine and not proof that any transaction is fraudulent.

## Buildathon challenge

**Track:** AI Risk Manager<br>
**Objective:** Build a working detector for a class of financial loss and measure precision and recall on a held-out test set.<br>
**Selected problem:** Credit-card transaction fraud detection and risk scoring.

## Product tour

- **Overview** — key held-out metrics, dataset facts, model snapshot, and decision pipeline.
- **Transaction Risk Checker** — raw Amount and V1–V28 input, real held-out demo examples, probability, score, category, and recommendation.
- **Risk Analytics** — fraud capture, false alerts, missed fraud, confusion-matrix outcomes, and exact risk bands.
- **Explainability** — actual fitted-model feature importance with non-causal interpretation.
- **Model Performance** — precision, recall, F1, PR-AUC, ROC-AUC, confusion matrix, PR curve, and ROC curve.
- **About** — approach, limitations, technology, and future improvements.

<!-- Add hosted application screenshots here after deployment. -->

## Architecture

```text
OpenML 1597
  → schema and data-quality validation
  → stratified, duplicate-group-aware train / validation / test split
  → deterministic feature engineering inside each model pipeline
  → training-fitted imputation/scaling
  → weighted Logistic Regression + weighted XGBoost
  → validation PR-AUC model selection
  → validation-only F1 threshold selection
  → one final held-out test evaluation
  → saved pipeline, metrics, figures, importance, and demo samples
  → Streamlit dashboard
```

The test labels are not used for model or threshold selection. Identical feature rows are grouped so they cannot cross split boundaries. The saved model owns feature engineering and preprocessing, so `model.predict_proba(raw_dataframe)` is the deployment contract.

## Dataset

The project downloads [OpenML dataset 1597](https://www.openml.org/d/1597), the anonymized ULB/Worldline credit-card fraud benchmark. It contains V1–V28, Amount, and Class. Some representations include Time; the pipeline treats it as optional.

- No merchant, customer, location, device, browser, or transaction-history features are invented.
- Missing values, numeric validity, duplicates, and class distribution are reported before training.
- The benchmark is severely imbalanced, so accuracy is not the primary metric.

Raw data is reproducibly downloaded and excluded from Git.

## Machine-learning pipeline

1. Validate schema, target, numeric values, missing values, duplicates, and label distribution.
2. Create deterministic Amount transforms: `log1p`, square root, zero flag, and squared Amount.
3. Create safe elapsed-time transforms only if Time is present.
4. Create 60/20/20 group-aware stratified train/validation/test splits with seed 42.
5. Fit preprocessing on training rows only.
6. Train class-weighted Logistic Regression and XGBoost using a training-derived imbalance weight.
7. Select the model by validation PR-AUC.
8. Select its threshold on validation data by maximum F1, breaking ties with higher recall.
9. Freeze the pipeline and threshold, then evaluate once on the held-out test set.

Generated metrics are stored in `reports/metrics/`; they are never fabricated or hardcoded in the app. Run training to refresh the metrics below and dashboard artifacts.

## Risk scoring

`Risk Score = model fraud probability × 100`

| Score | Category | Guidance |
|---:|---|---|
| 0–29.99 | LOW | Low model-estimated fraud risk. |
| 30–59.99 | MEDIUM | Review may be appropriate. |
| 60–79.99 | HIGH | Additional verification recommended. |
| 80–100 | CRITICAL | Strong model-estimated signal; consider blocking or enhanced review. |

The independently selected fraud-decision threshold determines whether a transaction is flagged for review; the risk band provides a human-readable severity layer.

## Explainability

The dashboard reads actual importance values from the selected fitted model. XGBoost uses its learned feature importance; Logistic Regression uses absolute fitted coefficients. Feature importance represents model influence and should not be interpreted as causation. The anonymized PCA features do not support assumed business meanings.

## Local installation

Python 3.11 or newer is recommended.

```powershell
git clone https://github.com/Kashish-13/ai-risk-manager.git
cd ai-risk-manager
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\download_data.py
python scripts\train_model.py
pytest
streamlit run app.py
```

On macOS/Linux, activate with `source .venv/bin/activate` and use forward slashes in script paths.

The repository includes the compact runtime model and report artifacts used by the hosted demo, so deployment does not retrain at startup. Downloading and retraining is needed only to reproduce or update them.

## Streamlit Community Cloud deployment

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create an app with:
   - Repository: `Kashish-13/ai-risk-manager`
   - Branch: `main`
   - Main file path: `app.py`
3. Deploy. No API key or secret is required.

The app uses repository-relative `pathlib` paths and contains no Windows-only runtime paths.

## Repository structure

```text
├── app.py                    # Streamlit entrypoint
├── .streamlit/config.toml    # cloud-safe dark theme
├── config/config.yaml        # data, model, split, and risk settings
├── data/                     # reproducible local dataset area
├── models/                   # deployable fitted pipeline and metadata
├── reports/
│   ├── figures/              # held-out and validation plots
│   └── metrics/              # genuine JSON results
├── scripts/
│   ├── download_data.py
│   └── train_model.py
├── src/                      # validation, features, training, evaluation, inference
├── tests/                    # focused automated contract tests
└── requirements.txt
```

## Limitations

- Historical public benchmark rather than live payment-network data.
- Anonymized PCA signals limit business-level interpretation.
- No merchant, account, customer, device, or geographic context.
- False positives and false negatives remain possible.
- Model drift, security, privacy, fairness, latency, monitoring, and threshold governance need production validation.

## Future improvements

- Temporal/out-of-time validation and probability calibration.
- Drift, data-quality, and live performance monitoring.
- Governed cost-sensitive threshold selection by business segment.
- Authorized merchant/customer context and human-review feedback loops.
- Secure API serving, authentication, audit logs, and production observability.

## License

See [LICENSE](LICENSE).
