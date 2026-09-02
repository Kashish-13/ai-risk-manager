from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.features import engineer_features


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "final_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics" / "test_metrics.json"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
THRESHOLD_PATH = ROOT / "models" / "threshold.json"
IMPORTANCE_PATH = ROOT / "reports" / "metrics" / "feature_importance.json"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Risk Manager",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
    }

    .risk-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        margin-bottom: 15px;
    }

    .small-text {
        color: #6b7280;
        font-size: 14px;
    }

    .risk-title {
        font-size: 28px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD FUNCTIONS
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# RISK FUNCTIONS
# ============================================================

def get_risk_level(score):
    if score >= 90:
        return "CRITICAL"
    elif score >= 70:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    return "LOW"


def get_risk_message(level):
    messages = {
        "LOW": "Transaction appears low risk.",
        "MEDIUM": "Transaction requires additional review.",
        "HIGH": "Transaction shows elevated fraud risk.",
        "CRITICAL": "Transaction should be investigated immediately.",
    }

    return messages[level]


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🛡️ AI Risk Manager")

st.sidebar.caption(
    "AI-powered transaction fraud detection"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Transaction Risk Checker",
        "Risk Analytics",
        "Explainability",
        "Model Performance",
        "About",
    ],
)

st.sidebar.divider()

st.sidebar.caption(
    "Fraud Risk Intelligence Platform"
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.title("🛡️ AI Risk Manager")

    st.caption(
        "AI-powered transaction fraud detection and risk scoring"
    )

    st.divider()

    metrics = load_json(METRICS_PATH)
    metadata = load_json(METADATA_PATH)

    # --------------------------------------------------------
    # PRIMARY METRICS
    # --------------------------------------------------------

    st.subheader("Risk Management Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Precision",
            f"{metrics['precision'] * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Recall",
            f"{metrics['recall'] * 100:.2f}%"
        )

    with col3:
        st.metric(
            "F1 Score",
            f"{metrics['f1'] * 100:.2f}%"
        )

    with col4:
        st.metric(
            "PR-AUC",
            f"{metrics['pr_auc'] * 100:.2f}%"
        )

    st.divider()

    # --------------------------------------------------------
    # TEST RESULTS
    # --------------------------------------------------------

    st.subheader("Held-out Test Results")

    confusion = np.array(
        metrics["confusion_matrix"]
    )

    true_negative = int(confusion[0][0])
    false_positive = int(confusion[0][1])
    false_negative = int(confusion[1][0])
    true_positive = int(confusion[1][1])

    total_transactions = (
        true_negative
        + false_positive
        + false_negative
        + true_positive
    )

    total_fraud = (
        false_negative
        + true_positive
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "ROC-AUC",
            f"{metrics['roc_auc']:.4f}"
        )

    with col2:
        st.metric(
            "False Positives",
            false_positive
        )

    with col3:
        st.metric(
            "False Negatives",
            false_negative
        )

    with col4:
        st.metric(
            "Fraud Detected",
            f"{true_positive} / {total_fraud}"
        )

    st.divider()

    # --------------------------------------------------------
    # MODEL STATUS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Current Model")

        st.success(
            "XGBoost"
        )

        st.write(
            """
            XGBoost was selected using validation PR-AUC.
            The fraud decision threshold was calibrated on
            validation data before final held-out testing.
            """
        )

    with col2:

        st.subheader("Decision Threshold")

        threshold_data = load_json(
            THRESHOLD_PATH
        )

        threshold = float(
            threshold_data["fraud_decision_threshold"]
        )

        st.metric(
            "Calibrated Threshold",
            f"{threshold * 100:.2f}%"
        )

        st.write(
            "Transactions above this model probability are flagged for review."
        )

    st.divider()

    st.info(
        f"""
        The held-out test set contains **{total_transactions:,} transactions**,
        including **{total_fraud} fraud cases**.

        Final evaluation is performed on data that was not used for
        model selection or threshold calibration.
        """
    )


# ============================================================
# TRANSACTION RISK CHECKER
# ============================================================

elif page == "Transaction Risk Checker":

    st.title("🔍 Transaction Risk Checker")

    st.caption(
        "Analyze an individual transaction using the trained fraud model."
    )

    st.divider()

    model = load_model()

    metadata = load_json(
        METADATA_PATH
    )

    threshold_data = load_json(
        THRESHOLD_PATH
    )

    threshold = float(
        threshold_data["fraud_decision_threshold"]
    )

    feature_columns = metadata[
        "raw_feature_columns"
    ]

    st.info(
        """
        The source dataset contains anonymized transaction features.
        Device ID, merchant location, account age and similar operational
        fields are not available in the dataset.
        """
    )

    # --------------------------------------------------------
    # AMOUNT
    # --------------------------------------------------------

    st.subheader("Transaction Details")

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0,
        step=10.0,
    )

    st.divider()

    # --------------------------------------------------------
    # ANONYMIZED FEATURES
    # --------------------------------------------------------

    with st.expander(
        "⚙️ Advanced Transaction Features",
        expanded=False,
    ):

        st.caption(
            "V1–V28 are anonymized features from the source dataset."
        )

        feature_values = {}

        anonymized_features = [
            column
            for column in feature_columns
            if column not in ["Amount", "Class"]
        ]

        columns = st.columns(2)

        for index, column in enumerate(
            anonymized_features
        ):

            with columns[index % 2]:

                feature_values[column] = st.number_input(
                    column,
                    value=0.0,
                    format="%.6f",
                    key=f"risk_input_{column}",
                )

    st.divider()

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    analyze = st.button(
        "🔎 Analyze Transaction",
        type="primary",
        use_container_width=True,
    )

    if analyze:

        row = {}

        for column in feature_columns:

            if column == "Amount":

                row[column] = amount

            elif column == "Class":

                continue

            else:

                row[column] = feature_values.get(
                    column,
                    0.0,
                )

        transaction = pd.DataFrame(
            [row]
        )

        input_columns = [
            column
            for column in feature_columns
            if column != "Class"
        ]

        transaction = transaction[
            input_columns
        ]

        engineered_transaction = engineer_features(
            transaction
        )

        probability = float(
            model.predict_proba(
                engineered_transaction
            )[0, 1]
        )

        risk_score = probability * 100

        risk_level = get_risk_level(
            risk_score
        )

        risk_message = get_risk_message(
            risk_level
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.divider()

        st.subheader("🎯 Risk Assessment")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Fraud Probability",
                f"{probability * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Risk Score",
                f"{risk_score:.2f} / 100"
            )

        with col3:

            st.metric(
                "Risk Level",
                risk_level
            )

        # ----------------------------------------------------
        # RISK MESSAGE
        # ----------------------------------------------------

        if risk_level == "CRITICAL":

            st.error(
                f"🚨 {risk_message}"
            )

        elif risk_level == "HIGH":

            st.warning(
                f"⚠️ {risk_message}"
            )

        elif risk_level == "MEDIUM":

            st.info(
                f"🔎 {risk_message}"
            )

        else:

            st.success(
                f"✅ {risk_message}"
            )

        # ----------------------------------------------------
        # DECISION
        # ----------------------------------------------------

        st.subheader("Decision")

        if probability >= threshold:

            st.error(
                "🚨 FLAG FOR FRAUD REVIEW"
            )

            st.write(
                f"""
                Model probability is **{probability * 100:.2f}%**,
                which is above the calibrated decision threshold
                of **{threshold * 100:.2f}%**.
                """
            )

        else:

            st.success(
                "✅ ALLOW / LOW-RISK TRANSACTION"
            )

            st.write(
                f"""
                Model probability is **{probability * 100:.2f}%**,
                which is below the calibrated decision threshold
                of **{threshold * 100:.2f}%**.
                """
            )

        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        st.subheader("🧠 Risk Interpretation")

        st.write(
            """
            The score represents the model's estimated fraud
            likelihood based on the available transaction features.

            Model signals should be treated as risk indicators,
            not as proof that a transaction is fraudulent.
            """
        )


# ============================================================
# RISK ANALYTICS
# ============================================================

elif page == "Risk Analytics":

    st.title("📈 Risk Analytics")

    st.caption(
        "Operational fraud detection metrics from the held-out test set."
    )

    st.divider()

    metrics = load_json(
        METRICS_PATH
    )

    confusion = np.array(
        metrics["confusion_matrix"]
    )

    true_negative = int(confusion[0][0])
    false_positive = int(confusion[0][1])
    false_negative = int(confusion[1][0])
    true_positive = int(confusion[1][1])

    total_transactions = (
        true_negative
        + false_positive
        + false_negative
        + true_positive
    )

    total_fraud = (
        false_negative
        + true_positive
    )

    fraud_rate = (
        total_fraud
        / total_transactions
    ) * 100

    detection_rate = (
        true_positive
        / total_fraud
    ) * 100

    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Test Transactions",
            f"{total_transactions:,}"
        )

    with col2:

        st.metric(
            "Fraud Transactions",
            f"{total_fraud:,}"
        )

    with col3:

        st.metric(
            "Fraud Rate",
            f"{fraud_rate:.3f}%"
        )

    with col4:

        st.metric(
            "Detection Rate",
            f"{detection_rate:.2f}%"
        )

    st.divider()

    # --------------------------------------------------------
    # OUTCOMES
    # --------------------------------------------------------

    st.subheader("Operational Outcomes")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "True Negatives",
            f"{true_negative:,}"
        )

    with col2:
        st.metric(
            "False Positives",
            false_positive
        )

    with col3:
        st.metric(
            "False Negatives",
            false_negative
        )

    with col4:
        st.metric(
            "True Positives",
            true_positive
        )

    st.divider()

    # --------------------------------------------------------
    # RISK FRAMEWORK
    # --------------------------------------------------------

    st.subheader("Risk Level Framework")

    risk_framework = pd.DataFrame(
        {
            "Risk Level": [
                "LOW",
                "MEDIUM",
                "HIGH",
                "CRITICAL",
            ],
            "Risk Score": [
                "0–39.99",
                "40–69.99",
                "70–89.99",
                "90–100",
            ],
            "Suggested Action": [
                "Allow / monitor",
                "Additional review",
                "Fraud review",
                "Immediate investigation",
            ],
        }
    )

    st.dataframe(
        risk_framework,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # --------------------------------------------------------
    # BUSINESS TRADEOFF
    # --------------------------------------------------------

    st.subheader("False Positive vs False Negative Trade-off")

    st.write(
        f"""
        The model generated **{false_positive} false positives** on the
        held-out test set. In a real payment system, false positives can
        create customer friction and unnecessary manual investigations.

        The model also produced **{false_negative} false negatives**,
        representing fraudulent transactions that were not detected.

        This highlights the central fraud-management trade-off:
        increasing fraud detection can sometimes increase legitimate
        transaction friction.
        """
    )

    st.info(
        "The prototype keeps both types of errors visible so a risk team "
        "can evaluate model performance from an operational perspective."
    )


# ============================================================
# EXPLAINABILITY
# ============================================================

elif page == "Explainability":

    st.title("🧠 Explainability")

    st.caption(
        "Model-associated feature importance for fraud-risk prediction."
    )

    st.divider()

    importance_data = load_json(
        IMPORTANCE_PATH
    )

    st.info(
        importance_data["wording"]
    )

    features = pd.DataFrame(
        importance_data["features"]
    )

    # --------------------------------------------------------
    # CLEAN FEATURE NAMES
    # --------------------------------------------------------

    features["feature"] = (
        features["feature"]
        .str.replace(
            "numeric__",
            "",
            regex=False,
        )
    )

    features["importance_percent"] = (
        features["importance"] * 100
    )

    features = features.sort_values(
        "importance",
        ascending=False,
    )

    # --------------------------------------------------------
    # TOP FEATURES
    # --------------------------------------------------------

    st.subheader(
        "Top Model Risk Indicators"
    )

    top_features = features.head(10).copy()

    display_features = top_features[
        [
            "feature",
            "importance_percent",
        ]
    ].copy()

    display_features.columns = [
        "Feature",
        "Model Importance (%)",
    ]

    display_features[
        "Model Importance (%)"
    ] = display_features[
        "Model Importance (%)"
    ].round(2)

    st.dataframe(
        display_features,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    st.subheader(
        "Top 10 Feature Importance"
    )

    chart_data = (
        top_features[
            [
                "feature",
                "importance_percent",
            ]
        ]
        .set_index("feature")
        .sort_values(
            "importance_percent"
        )
    )

    st.bar_chart(
        chart_data
    )

    st.divider()

    # --------------------------------------------------------
    # TOP FEATURE
    # --------------------------------------------------------

    top_feature = top_features.iloc[0]

    st.subheader(
        "Model Signal"
    )

    st.write(
        f"""
        **{top_feature['feature']}** is the strongest model-associated
        feature in the reported feature-importance results, contributing
        approximately **{top_feature['importance_percent']:.2f}%**
        of the model's reported feature importance.
        """
    )

    st.warning(
        """
        Feature importance does not mean that a feature causes fraud.

        V1–V28 are anonymized variables in the source dataset, so their
        real-world business meanings are not assumed.
        """
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title("📊 Model Performance")

    st.caption(
        "Final performance on the held-out test dataset."
    )

    st.divider()

    metrics = load_json(
        METRICS_PATH
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    st.subheader(
        "Classification Metrics"
    )

    metrics_table = pd.DataFrame(
        {
            "Metric": [
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC",
                "PR-AUC",
            ],
            "Score": [
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
                metrics["roc_auc"],
                metrics["pr_auc"],
            ],
        }
    )

    metrics_table["Score"] = (
        metrics_table["Score"].round(4)
    )

    st.dataframe(
        metrics_table,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader(
        "Confusion Matrix"
    )

    cm = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[
            "Actual Legitimate",
            "Actual Fraud",
        ],
        columns=[
            "Predicted Legitimate",
            "Predicted Fraud",
        ],
    )

    st.dataframe(
        cm,
        use_container_width=True,
    )

    st.divider()

    # --------------------------------------------------------
    # AVAILABLE EVALUATION FIGURES
    # --------------------------------------------------------

    figures_dir = (
        ROOT
        / "reports"
        / "figures"
    )

    figure_files = [
        (
            "Confusion Matrix",
            figures_dir / "confusion_matrix.png",
        ),
        (
            "Precision–Recall Curve",
            figures_dir / "precision_recall_curve.png",
        ),
        (
            "ROC Curve",
            figures_dir / "roc_curve.png",
        ),
        (
            "Validation Model Comparison",
            figures_dir / "model_comparison.png",
        ),
        (
            "Validation Threshold Calibration",
            figures_dir / "risk_threshold_analysis.png",
        ),
    ]

    for title, path in figure_files:

        if path.exists():

            st.subheader(title)

            st.image(
                str(path),
                use_container_width=True,
            )

    st.divider()

    # --------------------------------------------------------
    # METHODOLOGY
    # --------------------------------------------------------

    st.subheader(
        "Evaluation Methodology"
    )

    st.write(
        """
        The dataset was divided into training, validation and held-out
        test sets.

        Model selection and fraud-threshold calibration were performed
        using validation data. The metrics displayed above are calculated
        on the held-out test set after the model and threshold were finalized.
        """
    )

    st.info(
        "Accuracy is not treated as the primary metric because fraud is "
        "highly imbalanced in the dataset."
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.title("ℹ️ About AI Risk Manager")

    st.caption(
        "Machine-learning based transaction fraud risk intelligence"
    )

    st.divider()

    st.subheader(
        "What is AI Risk Manager?"
    )

    st.write(
        """
        AI Risk Manager is a fraud detection prototype that uses machine
        learning to estimate the likelihood that a transaction may be
        fraudulent.

        The model probability is converted into a model-derived risk score
        from 0 to 100 and mapped to LOW, MEDIUM, HIGH and CRITICAL risk levels.
        """
    )

    st.subheader(
        "Technology Stack"
    )

    st.markdown(
        """
        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        - XGBoost
        - Streamlit
        - Joblib
        """
    )

    st.subheader(
        "Risk Workflow"
    )

    st.markdown(
        """
        **Transaction → Feature Engineering → XGBoost → Fraud Probability
        → Risk Score → Risk Level → Decision**
        """
    )

    st.subheader(
        "Dataset Limitation"
    )

    st.warning(
        """
        The underlying dataset contains anonymized transaction features.

        It does not provide operational fields such as device identity,
        merchant location, account age or transaction history.

        Therefore, the application does not claim to use unavailable
        features.
        """
    )

    st.subheader(
        "Important Disclaimer"
    )

    st.info(
        """
        This is a decision-support prototype.

        A high risk score indicates elevated model-estimated fraud risk;
        it does not prove that a transaction is fraudulent.
        """
    )

    st.divider()

    st.caption(
        "AI Risk Manager • Fraud Detection Prototype"
    )