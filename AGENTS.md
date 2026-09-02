# AI Risk Manager contributor guide

- Use Python 3.11+ and `pathlib.Path`; never embed machine-specific paths.
- Preserve scientific honesty: synthetic data must be visibly labelled and never reported as real data.
- Prevent leakage. Fit imputers, scalers, encoders, feature statistics, and model-selection logic using training data only.
- Only engineer features supported by the input schema. Never infer device, location, merchant, account-age, or failed-attempt fields from the ULB dataset.
- Keep the test set untouched until final evaluation. Prefer PR-AUC, precision, recall, and F1 to accuracy.
- Add or update focused tests for changed behavior. Run `pytest` and an import check before handing work off.
- Keep raw data, trained models, generated reports, and secrets out of Git.

