"""Non-causal, model-specific feature-indicator utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ranked_feature_importance(model_pipeline, *, top_n: int = 20) -> pd.DataFrame:
    """Return ranked model associations, not causal explanations."""
    preprocessor = model_pipeline.named_steps["preprocessing"]
    classifier = model_pipeline.named_steps["classifier"]
    names = preprocessor.get_feature_names_out()
    if hasattr(classifier, "coef_"):
        values = np.abs(np.asarray(classifier.coef_).ravel())
        source = "absolute_logistic_coefficient"
    elif hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_)
        source = "model_feature_importance"
    else:
        raise ValueError("Classifier does not expose coefficient or feature importance values.")
    result = pd.DataFrame({"feature": names, "importance": values, "importance_source": source})
    return result.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
