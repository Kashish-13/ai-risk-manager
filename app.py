"""Premium Streamlit interface for the trained AI Risk Manager."""
from __future__ import annotations

import html
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.config import load_config
from src.predict import validate_prediction_input
from src.risk_engine import probability_to_risk

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
METRICS = ROOT / "reports" / "metrics"
FIGURES = ROOT / "reports" / "figures"
PAGES = ["Overview", "Transaction Risk Checker", "Risk Analytics", "Explainability", "Model Performance", "About"]

st.set_page_config(page_title="AI Risk Manager", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root{--bg:#070a0f;--surface:#0e141d;--surface-2:#121a25;--border:rgba(255,255,255,.075);--text:#f4f7fb;--muted:#8e9aad;--red:#ff5b5b;--purple:#8b73ff;--green:#38d9a9;--amber:#f8c15c;--orange:#ff8a4c}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}.stApp{background:radial-gradient(900px 480px at 85% -10%,rgba(130,78,255,.13),transparent 62%),radial-gradient(700px 420px at 10% 0,rgba(255,91,91,.08),transparent 65%),var(--bg);color:var(--text)}
[data-testid="stHeader"]{background:transparent;height:2rem}#MainMenu,footer{visibility:hidden}.block-container{max-width:1240px;padding-top:1.5rem;padding-bottom:3rem}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0c1119,#080c12);border-right:1px solid var(--border)}[data-testid="stSidebar"]>div{padding-top:1.25rem}
.brand{padding:12px 8px 22px}.brand-mark{width:38px;height:38px;border-radius:12px;display:grid;place-items:center;background:linear-gradient(135deg,var(--red),#c9498d);box-shadow:0 8px 28px rgba(255,91,91,.25);font:800 19px Manrope;color:white;margin-bottom:14px}.brand-name{font:800 17px Manrope;letter-spacing:.07em}.brand-sub{color:var(--muted);font-size:12px;margin-top:3px}
.online{display:flex;align-items:center;gap:9px;margin:10px 8px 20px;padding:10px 12px;border:1px solid rgba(56,217,169,.16);background:rgba(56,217,169,.055);border-radius:12px;color:#a7ead7;font-size:11px;font-weight:700;letter-spacing:.08em}.online-dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 0 5px rgba(56,217,169,.09)}
[data-testid="stSidebar"] [role="radiogroup"]{gap:5px}[data-testid="stSidebar"] label{border:1px solid transparent;border-radius:11px;padding:7px 10px;transition:.2s}[data-testid="stSidebar"] label:hover{background:rgba(255,255,255,.035);border-color:var(--border)}[data-testid="stSidebar"] label:has(input:checked){background:linear-gradient(90deg,rgba(255,91,91,.14),rgba(139,115,255,.06));border-color:rgba(255,91,91,.18)}
h1,h2,h3{font-family:'Manrope',sans-serif;letter-spacing:-.035em}.eyebrow{font-size:11px;color:#ff7979;font-weight:800;letter-spacing:.16em;text-transform:uppercase;margin-bottom:10px}.page-head{margin:5px 0 25px}.page-head h1{font-size:clamp(2rem,4vw,3.35rem);line-height:1.08;margin:0 0 9px}.page-head p{color:var(--muted);font-size:1rem;margin:0;max-width:700px}
.hero{position:relative;overflow:hidden;padding:clamp(28px,5vw,62px);border:1px solid var(--border);border-radius:28px;background:linear-gradient(125deg,rgba(20,28,41,.97),rgba(11,17,26,.96));box-shadow:0 30px 90px rgba(0,0,0,.25);margin-bottom:22px}.hero:after{content:"";position:absolute;width:360px;height:360px;border-radius:50%;right:-110px;top:-150px;background:radial-gradient(circle,rgba(139,115,255,.22),rgba(255,91,91,.07) 44%,transparent 68%)}.hero h1{font-size:clamp(2.65rem,6vw,5.15rem);line-height:.98;max-width:880px;margin:10px 0 20px}.hero p{font-size:clamp(.98rem,2vw,1.16rem);line-height:1.7;color:#aab4c2;max-width:740px;margin:0}.hero-line{width:48px;height:3px;border-radius:4px;background:linear-gradient(90deg,var(--red),var(--purple));margin-bottom:22px}
.section-title{display:flex;align-items:center;gap:12px;margin:34px 0 14px;color:#dce3ec;font:700 12px Manrope;letter-spacing:.13em;text-transform:uppercase}.section-title:after{content:"";height:1px;background:var(--border);flex:1}
.kpi-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px}.kpi-grid.five{grid-template-columns:repeat(5,minmax(0,1fr))}.kpi{position:relative;overflow:hidden;min-height:142px;padding:19px;border:1px solid var(--border);border-radius:17px;background:linear-gradient(145deg,rgba(19,27,39,.98),rgba(13,19,28,.98));box-shadow:0 12px 34px rgba(0,0,0,.13);transition:transform .2s,border-color .2s}.kpi:hover{transform:translateY(-3px);border-color:rgba(255,91,91,.25)}.kpi:before{content:"";position:absolute;top:0;left:19px;width:36px;height:2px;background:var(--accent,var(--red))}.kpi-label{color:#8e9aad;font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase}.kpi-value{font:800 clamp(1.45rem,2.7vw,2.1rem) Manrope;color:#f8fafc;margin:18px 0 7px;white-space:nowrap}.kpi-note{color:#6f7c8f;font-size:11px;line-height:1.35}.kpi-icon{position:absolute;right:16px;top:14px;color:var(--accent,var(--red));font-size:15px;opacity:.75}
.mini-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.mini{padding:17px 18px;border:1px solid var(--border);border-radius:15px;background:rgba(16,23,33,.8)}.mini-label{color:#7f8b9c;font-size:11px;text-transform:uppercase;letter-spacing:.08em}.mini-value{font:750 1.35rem Manrope;margin-top:9px;color:#eef3f9}
.pipeline{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.pipe{position:relative;padding:21px 18px;border-radius:17px;border:1px solid var(--border);background:linear-gradient(145deg,#111925,#0d131c)}.pipe:not(:last-child):after{content:"›";position:absolute;right:-10px;top:36%;z-index:2;width:20px;height:20px;display:grid;place-items:center;border-radius:50%;background:#1d2633;color:#ff7777;font-weight:800}.pipe-num{color:#ff6c6c;font:800 11px Manrope;letter-spacing:.1em}.pipe h4{font:700 16px Manrope;margin:12px 0 5px}.pipe p{color:#7f8b9c;font-size:12px;margin:0}
.feature-row{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.feature{padding:19px;border:1px solid var(--border);border-radius:16px;background:#0e151f}.feature b{display:block;font:700 14px Manrope;margin-bottom:7px}.feature span{color:#7e8a9b;font-size:12px;line-height:1.45}
.sample-card{padding:16px 18px;border-radius:16px;border:1px solid var(--border);background:linear-gradient(120deg,rgba(255,91,91,.06),rgba(139,115,255,.045));margin-bottom:14px}.sample-card b{font:700 14px Manrope}.sample-card p{color:var(--muted);font-size:12px;margin:5px 0 0}
[data-testid="stForm"]{border:1px solid var(--border);border-radius:20px;background:rgba(14,20,29,.82);padding:22px}[data-testid="stNumberInput"] input{border-radius:11px;background:#0a1018;border-color:rgba(255,255,255,.09)}div[data-testid="stExpander"]{border:1px solid var(--border);border-radius:14px;background:#0b1119;margin-top:9px}
.stButton button,[data-testid="stFormSubmitButton"] button{min-height:46px;border-radius:12px!important;font-weight:750!important;border:1px solid rgba(255,91,91,.75)!important;background:linear-gradient(100deg,#f45158,#d94d75)!important;color:white!important;box-shadow:0 9px 25px rgba(255,91,91,.17);transition:transform .2s,box-shadow .2s}.stButton button:hover,[data-testid="stFormSubmitButton"] button:hover{transform:translateY(-2px);box-shadow:0 12px 30px rgba(255,91,91,.27)}
.result{display:grid;grid-template-columns:minmax(230px,.78fr) 1.22fr;border:1px solid var(--risk-border);border-radius:24px;overflow:hidden;background:linear-gradient(135deg,var(--risk-bg),#0c131d);box-shadow:0 25px 70px rgba(0,0,0,.22);margin:18px 0}.score-side{padding:30px;border-right:1px solid var(--border)}.score-label{color:#8f9bad;font-size:11px;letter-spacing:.14em;font-weight:800}.score{font:800 clamp(3.5rem,8vw,5.8rem) Manrope;line-height:1;color:var(--risk);margin:13px 0 4px}.score small{font-size:16px;color:#8490a0}.risk-track{height:8px;border-radius:99px;background:#26303c;overflow:hidden;margin-top:20px}.risk-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,#38d9a9,#f8c15c,var(--risk))}.result-side{padding:30px}.risk-pill{display:inline-block;padding:7px 11px;border-radius:99px;color:var(--risk);background:color-mix(in srgb,var(--risk) 12%,transparent);border:1px solid color-mix(in srgb,var(--risk) 28%,transparent);font-size:11px;font-weight:800;letter-spacing:.1em}.result-side h3{font-size:1.45rem;margin:18px 0 8px}.result-side p{color:#9aa5b5;line-height:1.55}.result-facts{display:grid;grid-template-columns:repeat(2,1fr);gap:9px;margin-top:18px}.fact{padding:13px;border-radius:12px;border:1px solid var(--border);background:rgba(4,8,13,.25)}.fact span{display:block;color:#738094;font-size:10px;text-transform:uppercase;letter-spacing:.07em}.fact b{display:block;margin-top:6px;font-size:13px}
.matrix-wrap{padding:18px;border:1px solid var(--border);border-radius:18px;background:#0e151f}.callout{padding:18px 20px;border:1px solid rgba(139,115,255,.2);border-left:3px solid var(--purple);border-radius:14px;background:rgba(139,115,255,.055);color:#a9b3c2;line-height:1.55}.callout b{color:#eee}
.chart-card{padding:15px;border:1px solid var(--border);border-radius:18px;background:#0e151f;margin-bottom:14px}.chart-card img{border-radius:12px}.chips{display:flex;flex-wrap:wrap;gap:8px}.chip{padding:7px 11px;border:1px solid var(--border);border-radius:99px;background:#111925;color:#bac4d1;font-size:12px}
@media(max-width:1050px){.kpi-grid.five{grid-template-columns:repeat(3,1fr)}}
@media(max-width:900px){.kpi-grid,.kpi-grid.five,.mini-grid{grid-template-columns:repeat(2,1fr)}.pipeline,.feature-row{grid-template-columns:repeat(2,1fr)}.pipe:nth-child(2):after{display:none}.result{grid-template-columns:1fr}.score-side{border-right:0;border-bottom:1px solid var(--border)}}
@media(max-width:600px){.block-container{padding:1rem}.kpi-grid,.kpi-grid.five,.mini-grid,.pipeline,.feature-row{grid-template-columns:1fr}.pipe:after{display:none}.hero{border-radius:20px}.result-facts{grid-template-columns:1fr}}
</style>
""",
    unsafe_allow_html=True,
)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required artifact is missing: {path.name}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read {path.name}. Restore the deployment artifacts.") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Invalid artifact format: {path.name}")
    return value


@st.cache_resource
def load_artifacts():
    try:
        return (
            joblib.load(MODELS / "final_model.joblib"),
            read_json(METRICS / "test_metrics.json"),
            read_json(MODELS / "model_metadata.json"),
            float(read_json(MODELS / "threshold.json")["fraud_decision_threshold"]),
            read_json(METRICS / "feature_importance.json"),
            read_json(MODELS / "demo_samples.json"),
            None,
        )
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        return None, {}, {}, 0.5, {}, {}, str(exc)


def section(label: str) -> None:
    st.markdown(f'<div class="section-title">{html.escape(label)}</div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str, eyebrow: str = "AI RISK MANAGER") -> None:
    st.markdown(
        f'<div class="page-head"><div class="eyebrow">{html.escape(eyebrow)}</div><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def kpi_cards(items: list[tuple[str, str, str, str, str]]) -> None:
    cards = "".join(
        f'<div class="kpi" style="--accent:{color}"><div class="kpi-icon">{icon}</div><div class="kpi-label">{html.escape(label)}</div><div class="kpi-value">{html.escape(value)}</div><div class="kpi-note">{html.escape(note)}</div></div>'
        for label, value, note, color, icon in items
    )
    grid_class = "kpi-grid five" if len(items) == 5 else "kpi-grid"
    st.markdown(f'<div class="{grid_class}">{cards}</div>', unsafe_allow_html=True)


def mini_cards(items: list[tuple[str, str]]) -> None:
    cards = "".join(f'<div class="mini"><div class="mini-label">{html.escape(label)}</div><div class="mini-value">{html.escape(value)}</div></div>' for label, value in items)
    st.markdown(f'<div class="mini-grid">{cards}</div>', unsafe_allow_html=True)


def dark_confusion_matrix(tn: int, fp: int, fn: int, tp: int):
    fig, ax = plt.subplots(figsize=(6.4, 4.5), facecolor="#0e151f")
    ax.set_facecolor("#0e151f")
    matrix = [[tn, fp], [fn, tp]]
    image = ax.imshow(matrix, cmap="magma", alpha=.84)
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{matrix[row][col]:,}", ha="center", va="center", color="white", fontsize=15, weight="bold")
    ax.set_xticks([0, 1], ["Legitimate", "Fraud"]); ax.set_yticks([0, 1], ["Legitimate", "Fraud"])
    ax.set_xlabel("Predicted", color="#8e9aad", labelpad=12); ax.set_ylabel("Actual", color="#8e9aad", labelpad=12)
    ax.tick_params(colors="#b8c2cf", length=0); [spine.set_visible(False) for spine in ax.spines.values()]
    fig.colorbar(image, ax=ax, fraction=.04, pad=.04).ax.tick_params(colors="#7f8b9c")
    fig.tight_layout()
    return fig


model, test_metrics, metadata, threshold, importance, demos, artifact_error = load_artifacts()

st.sidebar.markdown('<div class="brand"><div class="brand-mark">AI</div><div class="brand-name">AI RISK MANAGER</div><div class="brand-sub">Fraud Intelligence Platform</div></div>', unsafe_allow_html=True)
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Overview"
page = st.sidebar.radio("Navigation", PAGES, key="nav_page", label_visibility="collapsed")
if artifact_error:
    st.sidebar.error("MODEL UNAVAILABLE")
else:
    st.sidebar.markdown('<div class="online"><span class="online-dot"></span> MODEL ONLINE</div>', unsafe_allow_html=True)
    st.sidebar.caption(f"{metadata['model']} · threshold {threshold:.4f}")

if artifact_error:
    page_header("Model artifacts unavailable", "The interface loaded safely, but prediction assets could not be read.")
    st.error(artifact_error)
    st.info("Restore the committed model and metrics artifacts, then restart the application.")
    st.stop()

tn, fp = map(int, test_metrics["confusion_matrix"][0])
fn, tp = map(int, test_metrics["confusion_matrix"][1])
test_total, test_fraud = tn + fp + fn + tp, tp + fn
capture = tp / test_fraud if test_fraud else 0.0

metric_items = [
    ("PR-AUC", f"{test_metrics['pr_auc']:.2%}", "Performance on the rare fraud class", "#ff5b5b", "◫"),
    ("ROC-AUC", f"{test_metrics['roc_auc']:.2%}", "Overall ranking quality", "#8b73ff", "◌"),
    ("Precision", f"{test_metrics['precision']:.2%}", "Accuracy of fraud alerts", "#38d9a9", "✓"),
    ("Recall", f"{test_metrics['recall']:.2%}", "Share of fraud cases detected", "#f8c15c", "↗"),
    ("F1 Score", f"{test_metrics['f1']:.2%}", "Balance of precision and recall", "#ff8a4c", "◇"),
]

if page == "Overview":
    st.markdown(
        '<div class="hero"><div class="hero-line"></div><div class="eyebrow">AI-POWERED FRAUD INTELLIGENCE</div><h1>Detect risk before it becomes loss.</h1><p>AI Risk Manager analyzes transaction patterns and produces an interpretable fraud-risk signal for faster and smarter transaction decisions.</p></div>',
        unsafe_allow_html=True,
    )
    cta1, cta2, _ = st.columns([1, 1.25, 3.3])
    cta1.button("Analyze Transaction", on_click=lambda: st.session_state.update(nav_page="Transaction Risk Checker"), width="stretch")
    cta2.button("View Model Performance", on_click=lambda: st.session_state.update(nav_page="Model Performance"), width="stretch")
    section("Live model performance")
    kpi_cards(metric_items)
    section("Model snapshot")
    mini_cards([("Model", metadata["model"]), ("Decision Threshold", f"{threshold:.6f}"), ("Held-out Transactions", f"{test_total:,}"), ("Held-out Fraud Cases", f"{test_fraud:,}")])
    section("Fraud intelligence pipeline")
    st.markdown(
        '<div class="pipeline"><div class="pipe"><div class="pipe-num">STEP 01</div><h4>Transaction</h4><p>Raw Amount and anonymized V-signals</p></div><div class="pipe"><div class="pipe-num">STEP 02</div><h4>ML Probability</h4><p>XGBoost estimates fraud likelihood</p></div><div class="pipe"><div class="pipe-num">STEP 03</div><h4>Risk Score</h4><p>Probability converted to a 0–100 signal</p></div><div class="pipe"><div class="pipe-num">STEP 04</div><h4>Risk Decision</h4><p>Clear review recommendation</p></div></div>',
        unsafe_allow_html=True,
    )
    section("Dataset overview")
    mini_cards([("Total Transactions", f"{metadata['dataset_rows']:,}"), ("Fraud Transactions", f"{metadata['dataset_fraud_count']:,}"), ("Fraud Rate", f"{metadata['dataset_fraud_count']/metadata['dataset_rows']:.3%}"), ("Test Fraud Cases", f"{test_fraud:,}")])
    mini_cards([("Detected Fraud", f"{tp} / {test_fraud}"), ("Fraud Capture Rate", f"{capture:.2%}"), ("False Alerts", f"{fp}"), ("Missed Fraud", f"{fn}")])
    section("Why AI risk management matters")
    st.markdown('<div class="callout">Effective fraud detection requires balancing two risks: missed fraud can lead to financial loss, while excessive false alerts create unnecessary customer friction. AI Risk Manager makes this trade-off visible through transparent risk scoring and model performance insights.</div>', unsafe_allow_html=True)

elif page == "Transaction Risk Checker":
    page_header("Transaction Risk Analysis", "Evaluate a transaction using the trained fraud detection model.", "LIVE RISK ENGINE")
    section("Demo samples")
    st.markdown('<div class="sample-card"><b>Use a verified held-out transaction</b><p>Both presets come from real test rows saved after final evaluation. They are used for demonstration only.</p></div>', unsafe_allow_html=True)
    low_button, high_button, _ = st.columns([1, 1, 2])
    if low_button.button("Load Low-Risk Sample", width="stretch"):
        for key, value in demos["low_risk"]["features"].items(): st.session_state[f"tx_{key}"] = float(value)
        st.session_state.demo_name = "Low-risk sample"
    if high_button.button("Load High-Risk Sample", width="stretch"):
        for key, value in demos["high_risk"]["features"].items(): st.session_state[f"tx_{key}"] = float(value)
        st.session_state.demo_name = "High-risk sample"
    if st.session_state.get("demo_name"): st.caption(f"Loaded: {st.session_state.demo_name}")
    raw_columns = metadata["raw_feature_columns"]
    for name in raw_columns:
        st.session_state.setdefault(f"tx_{name}", 100.0 if name == "Amount" else 0.0)
    section("Transaction details")
    with st.form("risk-analysis-form"):
        st.markdown("#### Transaction amount")
        amount = st.number_input("Amount", min_value=0.0, format="%.4f", key="tx_Amount", help="Transaction value from the OpenML dataset schema.")
        groups = [("Signal group 01 · V1–V7", range(1, 8)), ("Signal group 02 · V8–V14", range(8, 15)), ("Signal group 03 · V15–V21", range(15, 22)), ("Signal group 04 · V22–V28", range(22, 29))]
        for label, numbers in groups:
            with st.expander(label):
                columns = st.columns(3)
                for index, number in enumerate(numbers):
                    name = f"V{number}"
                    with columns[index % 3]: st.number_input(name, format="%.6f", key=f"tx_{name}")
        if "Time" in raw_columns: st.session_state.setdefault("tx_Time", 0.0); st.caption("Time is passed from the transaction record and is not exposed as a demo control.")
        submitted = st.form_submit_button("ANALYZE TRANSACTION", width="stretch")
    if submitted:
        try:
            values = {name: float(st.session_state[f"tx_{name}"]) for name in raw_columns}
            with st.spinner("Analyzing transaction risk..."):
                row = validate_prediction_input(pd.DataFrame([values]), raw_columns)
                probability = float(model.predict_proba(row)[0, 1])
                result = probability_to_risk(probability, load_config().risk_thresholds)
            risk_colors = {"LOW":"#38d9a9", "MEDIUM":"#f8c15c", "HIGH":"#ff8a4c", "CRITICAL":"#ff5b5b"}
            risk_backgrounds = {"LOW":"rgba(56,217,169,.07)", "MEDIUM":"rgba(248,193,92,.07)", "HIGH":"rgba(255,138,76,.07)", "CRITICAL":"rgba(255,91,91,.08)"}
            messages = {"LOW":"The model estimates low fraud risk.", "MEDIUM":"The model indicates that review may be appropriate.", "HIGH":"The model estimates elevated fraud risk and recommends additional verification.", "CRITICAL":"The model estimates a strong fraud signal; enhanced review or blocking may be appropriate."}
            decision = "FLAG FOR REVIEW" if probability >= threshold else "BELOW FRAUD THRESHOLD"
            st.markdown(
                f'<div class="result" style="--risk:{risk_colors[result.category]};--risk-bg:{risk_backgrounds[result.category]};--risk-border:{risk_colors[result.category]}55"><div class="score-side"><div class="score-label">RISK SCORE</div><div class="score">{result.score:.1f}<small>/100</small></div><div class="risk-track"><div class="risk-fill" style="width:{result.score:.2f}%"></div></div></div><div class="result-side"><div class="risk-pill">{result.category} RISK</div><h3>Model Risk Assessment</h3><p>{messages[result.category]}</p><div class="result-facts"><div class="fact"><span>Fraud Probability</span><b>{probability:.2%}</b></div><div class="fact"><span>Decision Threshold</span><b>{threshold:.2%}</b></div><div class="fact"><span>Model Used</span><b>{html.escape(metadata["model"])}</b></div><div class="fact"><span>Model Decision</span><b>{decision}</b></div></div></div></div>',
                unsafe_allow_html=True,
            )
            st.caption("This output is a model-estimated decision-support signal, not proof that fraud occurred.")

            threshold_gap = probability - threshold
            threshold_gap_pp = abs(threshold_gap) * 100
            threshold_position = "above" if threshold_gap >= 0 else "below"

            recommended_actions = {
                "LOW": "Allow normal processing and continue standard monitoring.",
                "MEDIUM": "Route for light manual review or step-up verification before final approval.",
                "HIGH": "Hold the transaction and perform additional verification before approval.",
                "CRITICAL": "Escalate immediately and block or decline pending investigation according to policy.",
            }

            section("Recommended action")
            st.markdown(
                f'<div class="callout"><b>{html.escape(result.category)} risk response</b><br>'
                f'{html.escape(recommended_actions[result.category])}</div>',
                unsafe_allow_html=True,
            )

            section("Investigation summary")
            mini_cards([
                ("Risk Category", result.category),
                ("Fraud Probability", f"{probability:.2%}"),
                ("Threshold Position", f"{threshold_gap_pp:.2f} pp {threshold_position}"),
                ("Model Decision", decision),
            ])

            st.markdown(
                f'<div class="callout"><b>Threshold context</b><br>'
                f'The model probability is {threshold_gap_pp:.2f} percentage points {threshold_position} '
                f'the frozen fraud decision threshold of {threshold:.2%}. '
                f'This comparison explains the model decision without changing the trained model or threshold.</div>',
                unsafe_allow_html=True,
            )

            section("Analysis timeline")
            st.markdown(
                '<div class="pipeline">'
                '<div class="pipe"><div class="pipe-num">01</div><h4>Validate input</h4><p>Amount and anonymized V-signals are checked against the saved schema.</p></div>'
                '<div class="pipe"><div class="pipe-num">02</div><h4>Estimate probability</h4><p>The saved XGBoost pipeline calculates fraud probability.</p></div>'
                '<div class="pipe"><div class="pipe-num">03</div><h4>Map risk</h4><p>Probability is converted into the existing 0–100 risk score and severity band.</p></div>'
                '<div class="pipe"><div class="pipe-num">04</div><h4>Support review</h4><p>The frozen threshold and risk band guide the recommended next action.</p></div>'
                '</div>',
                unsafe_allow_html=True,
            )

            with st.expander("Investigation details"):
                st.write(f"Transaction amount: {amount:.4f}")
                st.write(f"Input features evaluated: {len(raw_columns)}")
                st.write(f"Fraud probability: {probability:.6f}")
                st.write(f"Frozen decision threshold: {threshold:.6f}")
                st.write(f"Threshold comparison: {threshold_gap_pp:.4f} percentage points {threshold_position}")
                if st.session_state.get("demo_name"):
                    st.write(f"Loaded demo: {st.session_state.demo_name}")
                st.caption("V1–V28 are anonymized PCA-derived signals; the interface does not assign business meanings to them.")

        except (ValueError, KeyError, TypeError) as exc:
            st.error(f"Unable to analyze this transaction: {exc}")

elif page == "Risk Analytics":
    page_header("Risk Analytics", "Operational outcomes from the untouched held-out test set.", "HELD-OUT EVALUATION")
    kpi_cards(metric_items)
    section("Confusion matrix summary")
    kpi_cards([("True Positives", f"{tp:,}", "Fraud correctly detected", "#38d9a9", "✓"), ("False Positives", f"{fp:,}", "Legitimate transactions flagged", "#f8c15c", "!"), ("True Negatives", f"{tn:,}", "Legitimate transactions cleared", "#8b73ff", "○"), ("False Negatives", f"{fn:,}", "Fraud cases missed", "#ff5b5b", "×")])
    chart_col, capture_col = st.columns([1.35, 1])
    with chart_col:
        section("Actual confusion matrix")
        st.markdown('<div class="matrix-wrap">', unsafe_allow_html=True); st.pyplot(dark_confusion_matrix(tn, fp, fn, tp), width="stretch"); st.markdown('</div>', unsafe_allow_html=True)
    with capture_col:
        section("Fraud capture")
        mini_cards([("Detected", f"{tp} / {test_fraud}"), ("Capture Rate", f"{capture:.2%}"), ("Missed Fraud", f"{fn}"), ("False Alerts", f"{fp}")])
        st.markdown('<div class="callout"><b>Why these metrics matter</b><br>Fraud is rare, so PR-AUC, precision, recall and capture rate reveal more than overall accuracy.</div>', unsafe_allow_html=True)

elif page == "Explainability":
    page_header("Model Explainability", "Understand which anonymized transaction signals most influence the model.", "INTERPRETABLE ML")
    section("Top risk indicators")
    features = pd.DataFrame(importance["features"]).head(12).copy()
    features["feature"] = features["feature"].str.replace("numeric__", "", regex=False)
    total_importance = features["importance"].sum()
    features["relative_importance"] = features["importance"] / total_importance * 100 if total_importance else 0
    ordered = features.sort_values("relative_importance")
    fig, ax = plt.subplots(figsize=(9.5, 5.4), facecolor="#0e151f"); ax.set_facecolor("#0e151f")
    bars = ax.barh(ordered["feature"], ordered["relative_importance"], color=["#6d5ce8" if value < ordered["relative_importance"].max() else "#ff5b5b" for value in ordered["relative_importance"]], height=.62)
    ax.bar_label(bars, fmt="%.1f%%", padding=5, color="#aeb8c5", fontsize=9); ax.tick_params(colors="#aeb8c5", length=0); ax.set_xlabel("Relative importance among displayed features", color="#7f8b9c", labelpad=12)
    ax.xaxis.grid(True, color="white", alpha=.055); ax.set_axisbelow(True); [spine.set_visible(False) for spine in ax.spines.values()]; fig.tight_layout()
    st.markdown('<div class="chart-card">', unsafe_allow_html=True); st.pyplot(fig, width="stretch"); st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="callout"><b>Important</b><br>Feature importance represents model influence and should not be interpreted as causation.</div>', unsafe_allow_html=True)
    section("How to interpret this view")
    st.markdown('<div class="feature-row"><div class="feature"><b>Anonymized inputs</b><span>V1–V28 are PCA-derived signals without disclosed business meanings.</span></div><div class="feature"><b>Relative influence</b><span>Higher importance means the fitted model relied more strongly on that signal globally.</span></div><div class="feature"><b>Not a reason code</b><span>Importance does not prove why a particular transaction was fraudulent.</span></div><div class="feature"><b>Decision support</b><span>Use importance with model performance, threshold context and human review.</span></div></div>', unsafe_allow_html=True)

elif page == "Model Performance":
    page_header("Model Performance", "Judge-ready evidence from the final held-out test evaluation.", "BUILDATHON RESULTS")
    kpi_cards(metric_items)
    mini_cards([("Threshold", f"{threshold:.6f}"), ("Held-out Transactions", f"{test_total:,}"), ("Fraud Detected", f"{tp} / {test_fraud}"), ("False Alerts", f"{fp}")])
    section("Evaluation curves")
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="chart-card"><div class="eyebrow">PRECISION–RECALL CURVE</div>', unsafe_allow_html=True)
        st.image(str(FIGURES / "precision_recall_curve.png"), width="stretch"); st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="chart-card"><div class="eyebrow">ROC CURVE</div>', unsafe_allow_html=True)
        st.image(str(FIGURES / "roc_curve.png"), width="stretch"); st.markdown('</div>', unsafe_allow_html=True)
    section("Confusion matrix")
    matrix_left, matrix_right = st.columns([1.2, 1])
    with matrix_left:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True); st.image(str(FIGURES / "confusion_matrix.png"), width="stretch"); st.markdown('</div>', unsafe_allow_html=True)
    with matrix_right:
        st.markdown(f'<div class="callout"><b>Frozen validation threshold</b><br>{threshold:.6f}<br><br>{html.escape(metadata["threshold_selection"]["selection_rule"])}</div>', unsafe_allow_html=True)
    section("Metric guide")
    st.markdown('<div class="feature-row"><div class="feature"><b>Precision</b><span>How many transactions flagged by the model were actually fraudulent.</span></div><div class="feature"><b>Recall</b><span>How many actual fraud cases were detected by the model.</span></div><div class="feature"><b>PR-AUC</b><span>A key ranking metric for highly imbalanced fraud detection datasets.</span></div><div class="feature"><b>ROC-AUC</b><span>How well the model separates fraud and legitimate cases across thresholds.</span></div></div>', unsafe_allow_html=True)

else:
    page_header("About AI Risk Manager", "A transparent fraud-risk portfolio project built for practical decision support.", "PROJECT PROFILE")
    section("Project summary")
    cards = [
        ("The Problem", "Rare fraud creates direct loss, chargebacks and difficult customer-friction trade-offs."),
        ("The Solution", "A detector that turns model probability into a transparent risk score and review recommendation."),
        ("How It Works", "Validated transaction inputs flow through one saved feature, preprocessing and XGBoost pipeline."),
        ("Dataset", "OpenML 1597 with anonymized V1–V28, Amount, optional Time and the Class label."),
        ("Model Approach", "Group-aware splits, weighted baselines, validation selection and one final held-out evaluation."),
        ("Risk Scoring", "Fraud probability × 100 mapped to LOW, MEDIUM, HIGH and CRITICAL severity bands."),
        ("Explainability", "Actual global fitted-model importance, described as influence rather than causation."),
        ("Limitations", "Historical anonymous data, no merchant context, possible drift, false alerts and missed fraud."),
    ]
    st.markdown('<div class="feature-row">' + "".join(f'<div class="feature"><b>{html.escape(title)}</b><span>{html.escape(body)}</span></div>' for title, body in cards) + '</div>', unsafe_allow_html=True)
    section("Technology stack")
    st.markdown('<div class="chips">' + "".join(f'<span class="chip">{name}</span>' for name in ["Python", "Streamlit", "XGBoost", "scikit-learn", "pandas", "NumPy", "OpenML"]) + '</div>', unsafe_allow_html=True)
    section("Project workflow")
    st.markdown('<div class="pipeline"><div class="pipe"><div class="pipe-num">01</div><h4>Data</h4><p>Validate source records</p></div><div class="pipe"><div class="pipe-num">02</div><h4>Features</h4><p>Engineer safe signals</p></div><div class="pipe"><div class="pipe-num">03</div><h4>Model</h4><p>Estimate probability</p></div><div class="pipe"><div class="pipe-num">04</div><h4>Decision</h4><p>Score and support review</p></div></div>', unsafe_allow_html=True)
    section("Future roadmap")
    st.markdown('<div class="callout"><b>Next stage</b><br>Temporal validation, probability calibration, drift monitoring, governed thresholds, secure APIs and human-review feedback loops.</div>', unsafe_allow_html=True)
    st.markdown('<div class="callout"><b>Intelligent decision support</b><br>Built as an intelligent decision-support platform, AI Risk Manager combines fraud detection, risk scoring, and model explainability to help identify high-risk transactions with greater speed and clarity.</div>', unsafe_allow_html=True)





