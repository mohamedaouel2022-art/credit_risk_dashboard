from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "credit_risk_model.joblib"
METRICS_PATH = APP_DIR / "model_metrics.json"
DEFAULT_DATA_PATH = APP_DIR / "credit_risk_dataset.csv"

st.set_page_config(
    page_title="CrediScope | Credit Risk Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#10233F"
NAVY_2 = "#18385F"
BLUE = "#3977F6"
TEAL = "#17A88A"
AMBER = "#E8A23A"
RED = "#D95368"
INK = "#17263C"
MUTED = "#6C7A90"
PALE = "#F4F7FB"
BORDER = "#E3E9F2"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root {{ --navy:{NAVY}; --blue:{BLUE}; --ink:{INK}; --muted:{MUTED}; --border:{BORDER}; }}
html, body, [class*="css"] {{ font-family:'DM Sans',sans-serif; color:var(--ink); }}
.stApp {{ background:#F7F9FC; }}
[data-testid="stHeader"] {{ background:rgba(247,249,252,.94); }}
[data-testid="stSidebar"] {{ background:#10233F; border-right:1px solid #223A59; }}
[data-testid="stSidebar"] * {{ color:#F3F6FB !important; }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{ color:#AEBED3 !important; }}
[data-testid="stSidebar"] .stRadio label {{ padding:8px 10px; border-radius:9px; }}
[data-testid="stSidebar"] .stRadio label:hover {{ background:#1A3558; }}
.block-container {{ padding-top:1.7rem; padding-bottom:3rem; max-width:1500px; }}
h1,h2,h3 {{ font-family:'Manrope',sans-serif !important; letter-spacing:-.035em; }}
h1 {{ font-weight:800 !important; color:#132844; }}
h2 {{ font-weight:750 !important; color:#132844; }}
h3 {{ font-weight:700 !important; color:#183451; }}
[data-testid="stMetric"] {{ background:white; border:1px solid var(--border); padding:17px 19px; border-radius:15px; box-shadow:0 3px 14px rgba(16,35,63,.035); }}
[data-testid="stMetricLabel"] {{ color:#687991; font-size:.83rem; }}
[data-testid="stMetricValue"] {{ color:#122A48; font-family:'Manrope',sans-serif; font-weight:800; }}
[data-testid="stMetricDelta"] {{ font-size:.78rem; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background:white; border:1px solid var(--border); border-radius:16px; padding:16px 18px; }}
.stButton > button, .stDownloadButton > button {{ border-radius:10px; min-height:2.7rem; font-weight:700; border:1px solid #D7E0ED; }}
.stButton > button[kind="primary"] {{ background:#1B4C91; border-color:#1B4C91; color:white; }}
.stButton > button[kind="primary"]:hover {{ background:#153D76; border-color:#153D76; }}
div[data-testid="stForm"] {{ background:white; border:1px solid var(--border); padding:20px; border-radius:16px; }}
div[data-testid="stAlert"] {{ border-radius:12px; }}
.cs-eyebrow {{ color:#5B78A1; font-size:.73rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; margin-bottom:7px; }}
.cs-subtitle {{ color:#6B7C93; font-size:1rem; margin-top:-7px; margin-bottom:22px; }}
.cs-panel-title {{ font-family:'Manrope',sans-serif; font-size:1.03rem; font-weight:800; color:#17314F; margin-bottom:2px; }}
.cs-small {{ color:#718097; font-size:.83rem; }}
.cs-pill {{ display:inline-block; padding:5px 10px; border-radius:100px; background:#EAF1FF; color:#2355A4; font-size:.74rem; font-weight:800; }}
.cs-brand {{ font-family:'Manrope',sans-serif; font-size:1.28rem; font-weight:800; letter-spacing:-.04em; color:white; }}
.cs-brand-sub {{ color:#AABBD1; font-size:.68rem; letter-spacing:.16em; font-weight:700; }}
.cs-rule {{ height:1px; background:#2B4260; margin:18px 0; }}
.cs-risk-card {{ padding:22px 24px; background:white; border:1px solid var(--border); border-radius:17px; }}
.cs-result-label {{ color:#75849A; text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; font-weight:800; }}
.cs-result-value {{ font-family:'Manrope',sans-serif; color:#142B49; font-size:2.25rem; line-height:1.2; font-weight:800; }}
footer {{ visibility:hidden; }}
</style>
""", unsafe_allow_html=True)


def load_metrics():
    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text())
        except Exception:
            return {}
    return {}


def load_model():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None


def load_data(uploaded_file=None):
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file), "Uploaded dataset"
        except Exception as exc:
            st.error(f"Could not read uploaded CSV: {exc}")
            return None, None
    if DEFAULT_DATA_PATH.exists():
        try:
            return pd.read_csv(DEFAULT_DATA_PATH), DEFAULT_DATA_PATH.name
        except Exception:
            return None, None
    return None, None


def money(value, currency="$"):
    try:
        return f"{currency}{float(value):,.0f}"
    except (ValueError, TypeError):
        return "—"


def section_header(eyebrow, title, subtitle):
    st.markdown(f'<div class="cs-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="cs-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def chart_layout(fig, height=320):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color=MUTED, size=11),
        title_font=dict(family="Manrope", size=14, color=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False, linecolor=BORDER, zeroline=False),
        yaxis=dict(gridcolor="#EDF1F6", zeroline=False),
    )
    return fig


def risk_band(prob):
    if prob < 0.10:
        return "Lower", TEAL, "The model assigns a relatively low default-risk score."
    if prob < 0.25:
        return "Moderate", BLUE, "The model assigns a moderate default-risk score."
    if prob < 0.45:
        return "Elevated", AMBER, "The model assigns an elevated default-risk score."
    return "High", RED, "The model assigns a high default-risk score."


# Sidebar brand and navigation
uploaded = None
with st.sidebar:
    st.markdown('<div style="padding:10px 4px 6px"><div class="cs-brand">◈ CrediScope</div><div class="cs-brand-sub">CREDIT RISK INTELLIGENCE</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="cs-rule"></div>', unsafe_allow_html=True)
    page = st.radio(
        "WORKSPACE",
        ["Executive Overview", "Credit Assessment", "Portfolio Analytics", "Model Intelligence", "Data Explorer"],
        label_visibility="visible",
    )
    st.markdown('<div class="cs-rule"></div>', unsafe_allow_html=True)
    st.markdown("**WORKSPACE STATUS**")
    model = load_model()
    metrics = load_metrics()
    if model is not None:
        st.success("Model artifact connected")
    else:
        st.warning("Model not trained yet")
    st.caption("Local prototype · No data is sent to a server")
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    with st.expander("Load data"):
        uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"], help="Optional. The app also looks for credit_risk_dataset.csv beside app.py.")

df, data_source = load_data(uploaded)

if df is not None:
    if "loan_status" in df.columns:
        df["loan_status"] = pd.to_numeric(df["loan_status"], errors="coerce")
else:
    st.session_state.setdefault("last_data_source", None)

# Global header
left, right = st.columns([4, 1])
with left:
    st.markdown('<div class="cs-eyebrow">RISK & DECISION INTELLIGENCE</div>', unsafe_allow_html=True)
with right:
    st.markdown(f'<div style="text-align:right;color:#6F8096;font-size:.78rem;padding-top:4px">● LOCAL WORKSPACE &nbsp; <span style="color:#17A88A">ACTIVE</span></div>', unsafe_allow_html=True)

if page == "Executive Overview":
    section_header("Executive workspace", "Credit risk overview", "A consolidated view of lending risk, portfolio composition, and model readiness.")
    if df is None:
        st.info("To activate the live dashboard, place **credit_risk_dataset.csv** in this project folder or upload it from the sidebar. You can still explore the interface while setting up the model.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Portfolio records", "—")
        c2.metric("Observed default rate", "—")
        c3.metric("Model ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}" if metrics.get("roc_auc") is not None else "—")
        st.markdown("### Your workspace")
        a, b = st.columns(2)
        with a:
            st.markdown("#### 01 · Assess a borrower")
            st.write("Capture borrower and loan characteristics, then generate a model-based risk score.")
        with b:
            st.markdown("#### 02 · Monitor portfolio risk")
            st.write("Explore default outcomes, loan sizes, borrower segments, and model validation metrics.")
    else:
        n = len(df)
        default_rate = df["loan_status"].mean() if "loan_status" in df.columns else np.nan
        avg_loan = df["loan_amnt"].mean() if "loan_amnt" in df.columns else np.nan
        total_exposure = df["loan_amnt"].sum() if "loan_amnt" in df.columns else np.nan
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Applications in dataset", f"{n:,}")
        k2.metric("Observed default rate", f"{default_rate:.1%}" if pd.notna(default_rate) else "—")
        k3.metric("Average loan amount", money(avg_loan))
        k4.metric("Total requested exposure", money(total_exposure))
        st.markdown("")
        left, right = st.columns([1.35, 1])
        with left:
            with st.container(border=True):
                st.markdown('<div class="cs-panel-title">Default outcomes</div><div class="cs-small">Observed target distribution in the loaded dataset</div>', unsafe_allow_html=True)
                if "loan_status" in df.columns:
                    counts = df["loan_status"].map({0: "Non-default", 1: "Default"}).fillna("Unknown").value_counts().rename_axis("Outcome").reset_index(name="Applications")
                    fig = px.bar(counts, x="Outcome", y="Applications", color="Outcome",
                                 color_discrete_map={"Non-default": BLUE, "Default": RED, "Unknown": "#A7B2C2"},
                                 text_auto=".2s")
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(chart_layout(fig, 310), use_container_width=True)
                else:
                    st.warning("The dataset does not contain a loan_status target column.")
        with right:
            with st.container(border=True):
                st.markdown('<div class="cs-panel-title">Loan amount distribution</div><div class="cs-small">Understand the shape of requested credit</div>', unsafe_allow_html=True)
                if "loan_amnt" in df.columns:
                    fig = px.histogram(df, x="loan_amnt", nbins=28, color_discrete_sequence=[BLUE])
                    fig.update_traces(marker_line_width=0)
                    st.plotly_chart(chart_layout(fig, 310), use_container_width=True)
                else:
                    st.info("Add a loan_amnt column to display this chart.")
        if "loan_intent" in df.columns and "loan_status" in df.columns:
            with st.container(border=True):
                st.markdown('<div class="cs-panel-title">Risk by loan purpose</div><div class="cs-small">Observed default rate by application segment</div>', unsafe_allow_html=True)
                segment = df.groupby("loan_intent", dropna=False)["loan_status"].agg(["mean", "count"]).reset_index()
                segment.columns = ["Loan purpose", "Default rate", "Applications"]
                segment["Default rate"] *= 100
                fig = px.bar(segment.sort_values("Default rate"), x="Default rate", y="Loan purpose", orientation="h",
                             text=segment.sort_values("Default rate")["Default rate"].map(lambda x: f"{x:.1f}%"),
                             color="Default rate", color_continuous_scale=["#91B8FF", "#2459AD"])
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(chart_layout(fig, 350), use_container_width=True)
        st.caption(f"Data source: {data_source}. These are descriptive statistics, not a live lender portfolio.")

elif page == "Credit Assessment":
    section_header("Underwriting workspace", "New credit assessment", "Enter application characteristics to generate a model-based score. Review all outputs before making any real lending decision.")
    if model is None:
        st.warning("The prediction model is not available yet. Run `python train_model.py` in the project folder after adding your dataset.")
        st.code("python -m pip install -r requirements.txt\npython train_model.py\nstreamlit run app.py", language="bash")
    else:
        expected = metrics.get("feature_columns", [])
        feature_types = metrics.get("feature_types", {})
        if not expected:
            st.error("The model metadata does not include feature columns. Re-run `python train_model.py`.")
        else:
            defaults = {
                "person_age": 30, "person_income": 60000, "person_home_ownership": "RENT",
                "person_emp_length": 5, "loan_intent": "PERSONAL", "loan_grade": "C",
                "loan_amnt": 10000, "loan_int_rate": 12.0, "loan_percent_income": 0.17,
                "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 5,
            }
            choices = {
                "person_home_ownership": ["RENT", "MORTGAGE", "OWN", "OTHER"],
                "loan_intent": ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
                "loan_grade": ["A", "B", "C", "D", "E", "F", "G"],
                "cb_person_default_on_file": ["N", "Y"],
            }
            display_names = {
                "person_age": "Borrower age", "person_income": "Annual income",
                "person_home_ownership": "Housing status", "person_emp_length": "Employment length (years)",
                "loan_intent": "Loan purpose", "loan_grade": "Loan grade",
                "loan_amnt": "Requested loan amount", "loan_int_rate": "Interest rate (%)",
                "loan_percent_income": "Loan-to-income ratio",
                "cb_person_default_on_file": "Historical default on file",
                "cb_person_cred_hist_length": "Credit history length (years)",
            }
            with st.form("assessment_form"):
                st.markdown("#### Borrower profile")
                cols = st.columns(3)
                values = {}
                for i, name in enumerate(expected):
                    with cols[i % 3]:
                        label = display_names.get(name, name.replace("_", " ").title())
                        if name in choices:
                            opts = choices[name]
                            default = defaults.get(name, opts[0])
                            index = opts.index(default) if default in opts else 0
                            values[name] = st.selectbox(label, opts, index=index, key=f"assessment_{name}")
                        elif feature_types.get(name) == "numeric":
                            default = defaults.get(name, 0.0)
                            if name == "person_age":
                                values[name] = st.number_input(label, min_value=18, max_value=100, value=int(default), step=1, key=f"assessment_{name}")
                            elif name in ("person_income", "loan_amnt"):
                                values[name] = st.number_input(label, min_value=0.0, max_value=10_000_000.0, value=float(default), step=1000.0, key=f"assessment_{name}")
                            elif name in ("person_emp_length", "cb_person_cred_hist_length"):
                                values[name] = st.number_input(label, min_value=0.0, max_value=80.0, value=float(default), step=1.0, key=f"assessment_{name}")
                            elif "rate" in name:
                                values[name] = st.number_input(label, min_value=0.0, max_value=100.0, value=float(default), step=0.25, key=f"assessment_{name}")
                            else:
                                values[name] = st.number_input(label, min_value=0.0, max_value=100.0, value=float(defaults.get(name, 0.1)), step=0.01, key=f"assessment_{name}")
                        else:
                            values[name] = st.text_input(label, value=str(defaults.get(name, "")), key=f"assessment_{name}")
                st.markdown("---")
                submitted = st.form_submit_button("Generate risk assessment", type="primary", use_container_width=True)
            if submitted:
                row = pd.DataFrame([{name: values.get(name) for name in expected}], columns=expected)
                try:
                    probability = float(model.predict_proba(row)[0][1])
                    band, color, explanation = risk_band(probability)
                    threshold = float(metrics.get("decision_threshold", 0.35))
                    predicted_default = probability >= threshold
                    st.session_state["last_assessment"] = {
                        "values": values, "probability": probability, "band": band,
                        "predicted_default": bool(predicted_default), "threshold": threshold,
                    }
                except Exception as exc:
                    st.error(f"Assessment failed: {exc}. Confirm that the model was trained on the current dataset schema.")
            result = st.session_state.get("last_assessment")
            if result:
                st.markdown("### Assessment result")
                r1, r2, r3 = st.columns([1.1, 1, 1])
                with r1:
                    st.markdown(f'<div class="cs-risk-card"><div class="cs-result-label">Model risk score</div><div class="cs-result-value" style="color:{risk_band(result["probability"])[1]}">{result["probability"]:.1%}</div><div style="margin-top:9px"><span class="cs-pill">{result["band"].upper()} RISK</span></div><p style="color:#718097;font-size:.85rem;margin-top:13px">{risk_band(result["probability"])[2]}</p></div>', unsafe_allow_html=True)
                with r2:
                    st.metric("Threshold used", f'{result["threshold"]:.0%}')
                    st.write("A configurable model classification threshold; not a universal lending policy.")
                with r3:
                    st.metric("Model classification", "Higher-risk flag" if result["predicted_default"] else "Lower-risk flag")
                    st.write("This label is based on the selected threshold, not an approval or rejection.")
                st.warning("Prototype output only. This score is not guaranteed to be a calibrated probability of default and must not be used as the sole basis for a real credit decision.")
                with st.expander("Review submitted application data"):
                    st.dataframe(pd.DataFrame([result["values"]]), use_container_width=True, hide_index=True)
                report = {"risk_score": result["probability"], "risk_band": result["band"], "threshold": result["threshold"], "higher_risk_flag": result["predicted_default"], "inputs": result["values"], "model_version": metrics.get("model_name", "unknown")}
                st.download_button("Download assessment summary (JSON)", json.dumps(report, indent=2), file_name="credit_assessment.json", mime="application/json")

elif page == "Portfolio Analytics":
    section_header("Portfolio monitoring", "Portfolio analytics", "Segment exposure and observed outcomes to identify concentrations and areas for deeper review.")
    if df is None:
        st.info("Load the credit risk CSV from the sidebar or place it next to app.py to activate portfolio analytics.")
    else:
        numeric = df.select_dtypes(include="number")
        if "loan_amnt" in df.columns:
            total = df["loan_amnt"].sum()
            avg = df["loan_amnt"].mean()
            median = df["loan_amnt"].median()
        else:
            total = avg = median = np.nan
        a, b, c, d = st.columns(4)
        a.metric("Records", f"{len(df):,}")
        b.metric("Total requested", money(total))
        c.metric("Average loan", money(avg))
        d.metric("Median loan", money(median))
        candidate_segments = [c for c in ["loan_intent", "loan_grade", "person_home_ownership", "cb_person_default_on_file"] if c in df.columns]
        selected_segment = st.selectbox("Segment portfolio by", candidate_segments or ["No categorical segments available"])
        lcol, rcol = st.columns(2)
        with lcol:
            with st.container(border=True):
                st.markdown('<div class="cs-panel-title">Exposure by segment</div><div class="cs-small">Sum of requested loan amounts</div>', unsafe_allow_html=True)
                if selected_segment in df.columns and "loan_amnt" in df.columns:
                    exposure = df.groupby(selected_segment, dropna=False)["loan_amnt"].sum().sort_values(ascending=False).reset_index()
                    exposure.columns = ["Segment", "Exposure"]
                    fig = px.bar(exposure, x="Segment", y="Exposure", color_discrete_sequence=[BLUE], text_auto=".2s")
                    st.plotly_chart(chart_layout(fig, 330), use_container_width=True)
                else:
                    st.info("Required columns are not available.")
        with rcol:
            with st.container(border=True):
                st.markdown('<div class="cs-panel-title">Observed default rate</div><div class="cs-small">Historical target rate by selected segment</div>', unsafe_allow_html=True)
                if selected_segment in df.columns and "loan_status" in df.columns:
                    rates = df.groupby(selected_segment, dropna=False)["loan_status"].agg(["mean", "count"]).reset_index()
                    rates.columns = ["Segment", "Default rate", "Applications"]
                    rates["Default rate"] *= 100
                    fig = px.bar(rates.sort_values("Default rate"), x="Segment", y="Default rate", color="Default rate",
                                 color_continuous_scale=["#9DBEFF", "#234F9B"], hover_data=["Applications"])
                    fig.update_layout(coloraxis_showscale=False, yaxis_title="Default rate (%)")
                    st.plotly_chart(chart_layout(fig, 330), use_container_width=True)
                else:
                    st.info("A loan_status target column is needed for historical default-rate analysis.")
        if "loan_amnt" in df.columns and "loan_status" in df.columns:
            with st.container(border=True):
                st.markdown('<div class="cs-panel-title">Loan size vs. outcome</div><div class="cs-small">Explore the distribution of loan amounts by observed outcome</div>', unsafe_allow_html=True)
                plot_df = df.copy()
                plot_df["Outcome"] = plot_df["loan_status"].map({0: "Non-default", 1: "Default"}).fillna("Unknown")
                fig = px.box(plot_df, x="Outcome", y="loan_amnt", color="Outcome",
                             color_discrete_map={"Non-default": BLUE, "Default": RED, "Unknown": "#9BA7B8"}, points=False)
                fig.update_layout(showlegend=False)
                st.plotly_chart(chart_layout(fig, 350), use_container_width=True)
        st.caption("Historical associations do not establish causality and may reflect bias or differences in the dataset.")

elif page == "Model Intelligence":
    section_header("Model governance", "Model intelligence", "Review validation evidence, decision-threshold behavior, and the boundaries of the current prototype.")
    if not metrics:
        st.info("No model evaluation report is available yet. Train a model using `python train_model.py`.")
        st.code("python train_model.py", language="bash")
    else:
        name = metrics.get("model_name", "Trained model")
        st.markdown(f"**Selected model:** `{name}` &nbsp; · &nbsp; **Validation date:** `{metrics.get('trained_at', 'not recorded')}`", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Holdout ROC-AUC", f'{metrics.get("roc_auc", float("nan")):.3f}' if metrics.get("roc_auc") is not None else "—")
        m2.metric("Accuracy", f'{metrics.get("accuracy", float("nan")):.1%}' if metrics.get("accuracy") is not None else "—")
        m3.metric("Default precision", f'{metrics.get("default_precision", float("nan")):.1%}' if metrics.get("default_precision") is not None else "—")
        m4.metric("Default recall", f'{metrics.get("default_recall", float("nan")):.1%}' if metrics.get("default_recall") is not None else "—")
        left, right = st.columns(2)
        with left:
            with st.container(border=True):
                st.markdown('<div class="cs-panel-title">Confusion matrix</div><div class="cs-small">Holdout predictions at the configured threshold</div>', unsafe_allow_html=True)
                cm = metrics.get("confusion_matrix")
                if cm and len(cm) == 2:
                    fig = go.Figure(data=go.Heatmap(z=cm, x=["Predicted non-default", "Predicted default"], y=["Actual non-default", "Actual default"],
                                                    text=cm, texttemplate="%{text}", colorscale=[[0, "#EAF1FF"], [1, "#285AA6"]], showscale=False))
                    fig.update_layout(xaxis_title="", yaxis_title="")
                    st.plotly_chart(chart_layout(fig, 330), use_container_width=True)
                else:
                    st.info("Confusion matrix will appear after model training.")
        with right:
            with st.container(border=True):
                st.markdown('<div class="cs-panel-title">Class-level performance</div><div class="cs-small">Precision, recall and F1 at the saved threshold</div>', unsafe_allow_html=True)
                class_metrics = metrics.get("class_metrics", {})
                if class_metrics:
                    cm_df = pd.DataFrame(class_metrics).T.reset_index().rename(columns={"index": "Class"})
                    melted = cm_df.melt(id_vars="Class", var_name="Metric", value_name="Score")
                    fig = px.bar(melted, x="Metric", y="Score", color="Class", barmode="group",
                                 color_discrete_map={"Non-default": BLUE, "Default": RED}, range_y=[0, 1])
                    st.plotly_chart(chart_layout(fig, 330), use_container_width=True)
                else:
                    st.info("Class-level metrics will appear after training.")
        with st.container(border=True):
            st.markdown('<div class="cs-panel-title">Validation checklist</div><div class="cs-small">Before using this prototype in a real lending environment</div>', unsafe_allow_html=True)
            checks = [
                ("Holdout evaluation", "Metrics are measured on held-out records, but external validation is still needed."),
                ("Probability calibration", "A high ROC-AUC does not prove the scores are calibrated default probabilities."),
                ("Fairness and compliance", "Evaluate disparate impact, proxy variables, local laws, and adverse-action explanation requirements."),
                ("Monitoring", "Track drift, realized outcomes, missingness, and performance over time."),
                ("Human governance", "Do not make or communicate binding lending decisions solely from this prototype."),
            ]
            for title, desc in checks:
                st.markdown(f"**{title}**  \n<span style='color:#718097;font-size:.88rem'>{desc}</span>", unsafe_allow_html=True)
                st.markdown("")
        with st.expander("Technical model report (JSON)"):
            st.json(metrics)

elif page == "Data Explorer":
    section_header("Data quality", "Dataset explorer", "Inspect schema, missingness, class balance, and sample records before modeling.")
    if df is None:
        st.info("Upload a CSV using the sidebar or place `credit_risk_dataset.csv` next to `app.py`.")
        st.markdown("Expected target: `loan_status` where 0 = non-default and 1 = default. Check the meaning of all fields in your source dataset before training.")
    else:
        a, b, c = st.columns(3)
        a.metric("Rows", f"{df.shape[0]:,}")
        b.metric("Columns", f"{df.shape[1]:,}")
        c.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        t1, t2, t3 = st.tabs(["Preview", "Missingness", "Schema"])
        with t1:
            st.dataframe(df.head(100), use_container_width=True, hide_index=True)
            st.download_button("Download loaded data", df.to_csv(index=False), file_name="credit_risk_data_export.csv", mime="text/csv")
        with t2:
            missing = df.isna().sum().rename("Missing values").reset_index().rename(columns={"index": "Column"})
            missing["Missing %"] = (missing["Missing values"] / max(len(df), 1) * 100).round(2)
            missing = missing.sort_values("Missing %", ascending=False)
            fig = px.bar(missing[missing["Missing values"] > 0], x="Missing %", y="Column", orientation="h", color_discrete_sequence=[AMBER])
            st.plotly_chart(chart_layout(fig, max(250, 28 * max(1, len(missing[missing["Missing values"] > 0])))), use_container_width=True)
            st.dataframe(missing, use_container_width=True, hide_index=True)
        with t3:
            schema = pd.DataFrame({"Column": df.columns, "Data type": [str(t) for t in df.dtypes], "Unique values": [df[c].nunique(dropna=True) for c in df.columns]})
            st.dataframe(schema, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown(
    '<div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:#8794A7;font-size:.76rem">'
    '<span>CrediScope · Credit Risk Intelligence</span>'
    '<span>Research prototype · Not a lending decision engine</span>'
    '</div>', unsafe_allow_html=True
)
