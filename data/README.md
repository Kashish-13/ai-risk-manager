# Data policy

Primary data is the public ULB/Worldline Credit Card Fraud benchmark downloaded from OpenML dataset 1597. Run `python scripts/download_data.py` to place it in `data/raw/`.

The ULB dataset has anonymized `V1`–`V28` columns plus `Time`, `Amount`, and `Class`. It does not contain device, location, merchant, account-age, or failed-transaction data; this project does not fabricate those features.

`scripts/generate_synthetic_data.py` is development-only. Its output is explicitly marked `SYNTHETIC_DEVELOPMENT_ONLY` and must not be used to state real-world or benchmark model performance.

