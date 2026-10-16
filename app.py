"""
Credit Risk App
---------------
A small web interface built on top of the credit risk analysis script.

Run it with:   streamlit run app.py

How the file is organised (read it top to bottom):
  1. Settings and helper functions (load + clean data, train models, strategy table)
  2. The sidebar (the controls on the left)
  3. Four tabs: Data, Model comparison, Acceptance strategy, Score a loan
"""
import io
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import anthropic
except ImportError:
    anthropic = None
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (classification_report, confusion_matrix, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

THEMES = {
    "Classic Bank": dict(ink="#0B2545", muted="#5B6B82", deep="#13427A", mid="#13427A", a1="#00A88F", a2="#0B8F7B",
        gold="#F5B400", red="#E5484D", line="#C9D3E0", glow="rgba(11,37,69,.18)", bg="#F4F7FB", card="#FFFFFF",
        side="#FFFFFF", hero="linear-gradient(120deg,#0B2545 0%,#13427A 55%,#00A88F 120%)", orbs="none"),
}
for _t in THEMES.values():
    _t.setdefault("head", "'Inter', 'Cairo', sans-serif")
T = THEMES["Classic Bank"]
NAVY, TEAL, GOLD, RED = T["ink"], T["a1"], T["gold"], T["red"]
plt.rcParams.update({
    "axes.prop_cycle": plt.cycler(color=[T["deep"], T["a2"], GOLD, RED]),
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": T["line"], "axes.labelcolor": NAVY, "axes.titleweight": "bold",
    "axes.titlecolor": NAVY, "xtick.color": T["muted"], "ytick.color": T["muted"],
    "grid.color": T["line"], "grid.alpha": .6, "axes.grid": True, "figure.facecolor": "none",
    "axes.facecolor": "none", "savefig.transparent": True, "font.size": 10, "legend.frameon": False,
})

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Dancing+Script:wght@600;700&family=Cairo:wght@400;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', 'Cairo', sans-serif; }
:root { color-scheme: light; }
h1, h2, h3, .stApp .hero h1, [data-testid="stSidebar"] h1, [data-testid="stMetricValue"] { font-family: @head@ !important; }
.stApp { background: @bg@; background-attachment: fixed; }
.stApp::before { content: ""; position: fixed; inset: 0; pointer-events: none; background: @orbs@; z-index: 0; }
.block-container { padding-top: 1.5rem; max-width: 1250px; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: @side@; backdrop-filter: blur(14px); border-right: 1px solid @line@; }
[data-testid="stSidebar"] h1 { color: @ink@; font-weight: 800; border-bottom: 3px solid @a1@; padding-bottom: .4rem; }
.hero { background: @hero@; border-radius: 24px; padding: 30px 36px; margin-bottom: 1.2rem;
  box-shadow: 0 14px 40px @glow@; border: 1px solid rgba(255,255,255,.5); }
.stApp .hero h1 { color: #FFFFFF !important; font-size: 2.1rem; font-weight: 800; margin: 0; letter-spacing: -.5px; text-shadow: 0 2px 18px rgba(255,255,255,.45); }
.stApp .hero p { color: #FFFFFF !important; opacity: .92; margin: .4rem 0 0 0; font-size: 1rem; }
.hero .pill { display: inline-block; background: rgba(255,255,255,.25); color: #FFFFFF; border: 1px solid rgba(255,255,255,.7);
  border-radius: 999px; padding: 2px 14px; font-size: .75rem; font-weight: 700; margin-bottom: .6rem; letter-spacing: .1em; }
[data-testid="stMetric"] { background: @card@; backdrop-filter: blur(12px); border-radius: 18px; padding: 16px 18px;
  border: 1px solid rgba(255,255,255,.8); border-left: 5px solid @a1@; box-shadow: 0 6px 22px @glow@; }
[data-testid="stMetricLabel"] p { color: @muted@ !important; font-weight: 600; text-transform: uppercase; font-size: .72rem; letter-spacing: .06em; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] > div { color: @ink@; font-weight: 800;
  overflow: visible !important; text-overflow: clip !important; white-space: nowrap !important;
  font-size: clamp(1.15rem, 1.9vw, 1.9rem) !important; }
[data-testid="stMetric"] { overflow: visible; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: @card@; backdrop-filter: blur(12px); padding: 6px; border-radius: 999px; box-shadow: 0 4px 18px @glow@; }
.stTabs [data-baseweb="tab"] { border-radius: 999px; transition: background .25s ease, color .25s ease, box-shadow .25s ease; outline: none; padding: 10px 14px; flex: 1 1 0; min-width: 0; justify-content: center; font-weight: 600; font-size: 1rem; }
.stTabs [aria-selected="true"] { background: @hero@; border-radius: 999px; box-shadow: 0 4px 14px @glow@; }
.stTabs [aria-selected="false"]:hover { background: @line@; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; height: 0 !important; background: transparent !important; }
.stApp .stTabs [aria-selected="true"] p { color: #FFFFFF !important; }
.stApp .stTabs [aria-selected="false"] p { color: @ink@ !important; }
h2, h3 { color: @ink@ !important; font-weight: 700 !important; }
.stDataFrame, [data-testid="stImage"], .stPyplot { border-radius: 18px; overflow: hidden; }
.stPyplot { background: @card@; padding: 8px; box-shadow: 0 6px 22px @glow@; }
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button,
[data-testid="stFileUploaderDropzone"] button { background: linear-gradient(120deg, @a1@, @mid@) !important; color: #FFFFFF !important;
  border: 0 !important; border-radius: 12px; font-weight: 700; padding: .55rem 1.3rem; box-shadow: 0 6px 18px @glow@; }
.stApp .stButton p, .stApp .stDownloadButton p, .stApp [data-testid="stFormSubmitButton"] p,
[data-testid="stFileUploaderDropzone"] button * { color: #FFFFFF !important; }
.decision { border-radius: 20px; padding: 20px 26px; margin-top: 1rem; color: #fff; font-weight: 700; font-size: 1.3rem; box-shadow: 0 10px 30px @glow@; }
.decision.ok { background: linear-gradient(120deg, @a2@, @a1@); }
.decision.no { background: linear-gradient(120deg, @red@, @gold@); }
.decision small { display: block; font-weight: 500; font-size: .85rem; opacity: .95; margin-top: 4px; }
.stApp [data-testid="stMarkdownContainer"] p, .stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stWidgetLabel"] p, .stApp [data-testid="stWidgetLabel"] label,
.stApp [data-testid="stCaptionContainer"] p, .stApp label, .stApp [data-testid="stRadio"] p { color: @ink@ !important; }
.stApp .hero p, .stApp .hero h1 { color: #FFFFFF !important; }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] small { color: @ink@ !important; }
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color: @muted@ !important; }
[data-testid="stFileUploaderDropzone"] { background: rgba(255,255,255,.6) !important; border: 1px dashed @line@ !important; border-radius: 14px; }
[data-testid="stFileUploaderDropzone"] *:not(button):not(button *) { color: @ink@ !important; }
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="base-input"] { background: #FFFFFF !important; border-color: @line@ !important; }
[data-baseweb="input"] input, [data-baseweb="select"] div { color: @ink@ !important; -webkit-text-fill-color: @ink@; }
[data-testid="stCode"], [data-testid="stCode"] pre, [data-testid="stCode"] code { background: rgba(255,255,255,.85) !important; color: @ink@ !important; }
[data-testid="stAlert"] p { color: @ink@ !important; }
/* ---------- motion ---------- */
html { scroll-behavior: smooth; }
::selection { background: @a1@; color: #FFFFFF; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: @line@; border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: @a1@; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: none; } }
@keyframes gradientShift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
@keyframes floaty { 0%, 100% { transform: translateY(0) scale(1); } 50% { transform: translateY(-14px) scale(1.07); } }
@keyframes popIn { 0% { opacity: 0; transform: scale(.92); } 60% { transform: scale(1.03); } 100% { opacity: 1; transform: scale(1); } }
@keyframes slideIn { from { opacity: 0; transform: translateX(-14px); } to { opacity: 1; transform: none; } }
.hero { background-size: 220% 220%; animation: gradientShift 14s ease infinite, fadeUp .7s ease both; overflow: hidden; }
.hero::before, .hero::after { content: ""; position: absolute; border-radius: 50%; background: rgba(255,255,255,.14); pointer-events: none; }
.hero::before { width: 220px; height: 220px; right: -50px; top: -80px; animation: floaty 9s ease-in-out infinite; }
.hero::after { width: 140px; height: 140px; right: 150px; bottom: -70px; animation: floaty 7s ease-in-out infinite reverse; }
[data-testid="stMetric"] { animation: fadeUp .6s ease both; transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease; }
[data-testid="stMetric"]:hover { transform: translateY(-6px); box-shadow: 0 18px 36px @glow@; }
[data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stMetric"] { animation-delay: .08s; }
[data-testid="stHorizontalBlock"] > div:nth-child(3) [data-testid="stMetric"] { animation-delay: .16s; }
[data-testid="stHorizontalBlock"] > div:nth-child(4) [data-testid="stMetric"] { animation-delay: .24s; }
.stPyplot, [data-testid="stDataFrame"] { animation: fadeUp .6s ease both; transition: transform .25s ease, box-shadow .25s ease; }
.stPyplot:hover { transform: translateY(-4px); box-shadow: 0 18px 36px @glow@; }
.stTabs [data-baseweb="tab-panel"] { animation: fadeUp .45s ease both; }
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button,
[data-testid="stFileUploaderDropzone"] button { transition: transform .2s ease, box-shadow .2s ease, filter .2s ease; }
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover { transform: translateY(-3px); box-shadow: 0 12px 26px @glow@; }
.stButton > button:active, .stDownloadButton > button:active, [data-testid="stFormSubmitButton"] > button:active { transform: scale(.96); }
.decision { animation: popIn .55s cubic-bezier(.2,.9,.3,1.2) both; }
[data-testid="stChatMessage"] { animation: slideIn .4s ease both; border-radius: 18px; }
[data-testid="stSidebar"] [data-testid="stSlider"] { transition: transform .2s ease; }
[data-testid="stSidebar"] [data-testid="stSlider"]:hover { transform: translateX(3px); }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }
/* ---------- advanced effects ---------- */
.stApp::before { inset: -15%; filter: blur(30px); animation: drift 24s ease-in-out infinite alternate;
  background: radial-gradient(circle at 20% 25%, rgba(0,168,143,.20), transparent 38%),
              radial-gradient(circle at 80% 70%, rgba(19,66,122,.22), transparent 40%),
              radial-gradient(circle at 60% 15%, rgba(245,180,0,.12), transparent 30%); }
@keyframes drift { from { transform: translate(-3%, -2%) rotate(0deg) scale(1); } to { transform: translate(3%, 3%) rotate(8deg) scale(1.12); } }
@keyframes shine { from { background-position: 150% 0; } to { background-position: -150% 0; } }
@keyframes ripple { to { transform: scale(2.6); opacity: 0; } }
.stApp .hero h1 { background: linear-gradient(100deg, #fff 0%, #fff 38%, rgba(255,255,255,.55) 50%, #fff 62%, #fff 100%);
  background-size: 250% 100%; -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
  color: transparent !important; text-shadow: none; animation: shine 6s linear infinite; }
[data-testid="stMetric"], .stPyplot { position: relative; }
[data-testid="stMetric"]::before, .stPyplot::before { content: ""; position: absolute; inset: 0; border-radius: inherit; pointer-events: none;
  opacity: 0; transition: opacity .3s ease; z-index: 1;
  background: radial-gradient(260px circle at var(--mx, 50%) var(--my, 50%), rgba(0,168,143,.16), transparent 70%); }
[data-testid="stMetric"]:hover::before, .stPyplot:hover::before { opacity: 1; }
[data-testid="stMetric"] { transition: transform .12s ease-out, box-shadow .25s ease; will-change: transform; }
#fx-glow { position: fixed; left: 0; top: 0; width: 300px; height: 300px; border-radius: 50%; pointer-events: none; z-index: 1;
  background: radial-gradient(circle, rgba(0,168,143,.16), transparent 65%); transition: transform .12s ease-out; }
#fx-canvas { position: fixed; inset: 0; width: 100vw; height: 100vh; pointer-events: none; z-index: 1; opacity: .5; }
#fx-progress { position: fixed; top: 0; left: 0; height: 3px; width: 0; z-index: 999999; pointer-events: none;
  background: linear-gradient(90deg, @a1@, @deep@); transition: width .1s linear; }
.fx-ripple { position: absolute; border-radius: 50%; background: rgba(255,255,255,.55); transform: scale(0);
  animation: ripple .65s ease-out forwards; pointer-events: none; }
.gauge-bar { height: 14px; border-radius: 999px; position: relative; background: linear-gradient(90deg, #E5484D, #F5B400, #00A88F); margin: 14px 0 6px; }
.gauge-marker { position: absolute; top: -6px; width: 6px; height: 26px; border-radius: 3px; background: @ink@; transform: translateX(-50%); transition: left .8s ease; box-shadow: 0 2px 8px @glow@; }
.gauge-score { font-size: 2.2rem; font-weight: 800; color: @ink@; line-height: 1.1; }
@media (max-width: 768px) {
  .block-container { padding: 1rem .8rem 3rem; }
  .hero { padding: 20px 18px; border-radius: 18px; }
  .stApp .hero h1 { font-size: 1.5rem; }
  .stApp .hero p { font-size: .9rem; }
  .stTabs [data-baseweb="tab-list"] { overflow-x: auto; border-radius: 18px; }
  .stTabs [data-baseweb="tab"] { flex: 0 0 auto; padding: 8px 14px; font-size: .9rem; }
  [data-testid="stMetric"] { padding: 12px 14px; }
  #fx-canvas, #fx-glow { display: none !important; }
  .stApp::before { animation: none; filter: none; }
}
.hero { position: relative; }
.hero .sig { position: absolute; right: 34px; bottom: 20px; font-family: 'Dancing Script', cursive;
  font-size: 1.9rem; font-weight: 700; color: #FFFFFF !important; opacity: .95; }
.footer { text-align: center; color: @muted@; margin-top: 2.5rem; padding: 1rem; border-top: 1px solid @line@; font-size: .85rem; }
.footer .sigtxt { font-family: 'Dancing Script', cursive; font-size: 1.25rem; font-weight: 700; color: @ink@; }
</style>
"""
for k, v in T.items():
    CSS = CSS.replace("@" + k + "@", v)
st.markdown(CSS, unsafe_allow_html=True)

DEFAULT_CSV = Path(__file__).parent / "credit_risk_dataset.csv"

REQUIRED_COLUMNS = [
    "person_age", "person_income", "person_home_ownership", "person_emp_length",
    "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate", "loan_status",
    "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length",
]

TR = {  # key: (English, Français, العربية)
    "controls": ("Controls", "Contrôles", "عناصر التحكم"),
    "data_source": ("Data source", "Source des données", "مصدر البيانات"),
    "src_orig": ("Original dataset", "Jeu de données d'origine", "مجموعة البيانات الأصلية"),
    "src_up": ("Upload my own file", "Importer mon fichier", "رفع ملفي الخاص"),
    "upload_help": ("Must have the same columns as the original dataset", "Doit avoir les mêmes colonnes que le jeu d'origine", "يجب أن يحتوي على الأعمدة نفسها الموجودة في المجموعة الأصلية"),
    "csv_label": ("Credit risk CSV", "CSV risque de crédit", "ملف CSV لمخاطر الائتمان"),
    "lbl_up": ("Uploaded file: {name}", "Fichier importé : {name}", "الملف المرفوع: {name}"),
    "lbl_orig": ("Original dataset (credit_risk_dataset.csv)", "Jeu de données d'origine (credit_risk_dataset.csv)", "المجموعة الأصلية (credit_risk_dataset.csv)"),
    "active": ("Active data: {label} · {n:,} rows", "Données actives : {label} · {n:,} lignes", "البيانات الحالية: {label} · {n:,} صف"),
    "effects": ("Visual effects", "Effets visuels", "المؤثرات البصرية"),
    "reset": ("Reset and reload data", "Réinitialiser et recharger", "إعادة التعيين وتحميل البيانات"),
    "missing": ("This file is missing these columns: {cols}", "Il manque ces colonnes : {cols}", "ينقص هذا الملف الأعمدة التالية: {cols}"),
    "cutoff": ("Default cutoff (reports and confusion)", "Seuil de défaut (rapports et confusion)", "عتبة التعثر (التقارير والالتباس)"),
    "cutoff_help": ("A loan is predicted as a default when its probability is above this value", "Un prêt est prédit en défaut lorsque sa probabilité dépasse cette valeur", "يُتوقَّع تعثر القرض عندما يتجاوز احتماله هذه القيمة"),
    "accept": ("Acceptance rate", "Taux d'acceptation", "معدل القبول"),
    "accept_help": ("Share of new loans you are willing to accept", "Part des nouveaux prêts que vous acceptez", "نسبة القروض الجديدة التي ترغب في قبولها"),
    "lgd": ("Loss given default", "Perte en cas de défaut", "الخسارة عند التعثر"),
    "lgd_help": ("Share of the loan amount lost when a borrower defaults", "Part du montant perdue quand l'emprunteur fait défaut", "نسبة مبلغ القرض التي تُفقد عند تعثر المقترض"),
    "pill": ("RISK ANALYTICS", "ANALYSE DU RISQUE", "تحليل المخاطر"),
    "title": ("Credit Risk Dashboard", "Tableau de bord du risque de crédit", "لوحة مخاطر الائتمان"),
    "subtitle": ("Default prediction, model comparison and loan acceptance strategy in one place.", "Prédiction des défauts, comparaison des modèles et stratégie d'acceptation des prêts au même endroit.", "توقّع التعثر ومقارنة النماذج واستراتيجية قبول القروض في مكان واحد."),
    "tab_data": ("Data", "Données", "البيانات"),
    "tab_models": ("Model comparison", "Comparaison des modèles", "مقارنة النماذج"),
    "tab_strategy": ("Acceptance strategy", "Stratégie d'acceptation", "استراتيجية القبول"),
    "tab_score": ("Score a loan", "Évaluer un prêt", "تقييم قرض"),
    "tab_chat": ("AI Chatbot", "Chatbot IA", "شات بوت ذكي"),
    "loans": ("Loans (after cleaning)", "Prêts (après nettoyage)", "القروض (بعد التنظيف)"),
    "default_rate": ("Default rate", "Taux de défaut", "معدل التعثر"),
    "avg_loan": ("Average loan", "Prêt moyen", "متوسط القرض"),
    "first_rows": ("First rows", "Premières lignes", "الصفوف الأولى"),
    "loan_amounts": ("Loan amounts", "Montants des prêts", "مبالغ القروض"),
    "loan_amount_x": ("Loan amount", "Montant du prêt", "مبلغ القرض"),
    "intent_status": ("Loan intent vs loan status", "Objet du prêt et statut", "غرض القرض وحالته"),
    "reports_at": ("Classification reports at a {cutoff:.2f} cutoff", "Rapports de classification, seuil {cutoff:.2f}", "تقارير التصنيف عند عتبة {cutoff:.2f}"),
    "lr": ("Logistic regression", "Régression logistique", "الانحدار اللوجستي"),
    "gbt": ("Gradient boosted tree", "Arbre à gradient boosté", "شجرة التعزيز التدريجي"),
    "roc_cal": ("ROC chart and calibration curve", "Courbe ROC et calibration", "منحنى ROC ومنحنى المعايرة"),
    "roc_title": ("ROC chart: probability of default", "Courbe ROC : probabilité de défaut", "منحنى ROC: احتمال التعثر"),
    "fallout": ("False positive rate", "Taux de faux positifs", "معدل الإيجابيات الكاذبة"),
    "sensitivity": ("Sensitivity", "Sensibilité", "الحساسية"),
    "cal_title": ("Calibration curve", "Courbe de calibration", "منحنى المعايرة"),
    "avg_pred": ("Average predicted probability", "Probabilité prédite moyenne", "متوسط الاحتمال المتوقع"),
    "frac_pos": ("Fraction of positives", "Fraction de positifs", "نسبة الحالات الإيجابية"),
    "if_accept": ("If you accept {rate:.0%} of new loans", "Si vous acceptez {rate:.0%} des nouveaux prêts", "إذا قبلت {rate:.0%} من القروض الجديدة"),
    "threshold": ("Probability threshold", "Seuil de probabilité", "عتبة الاحتمال"),
    "bad_rate": ("Bad rate", "Taux de mauvais prêts", "معدل القروض السيئة"),
    "est_value": ("Estimated value", "Valeur estimée", "القيمة المقدّرة"),
    "exp_loss_acc": ("Expected loss (accepted)", "Perte attendue (acceptés)", "الخسارة المتوقعة (المقبولة)"),
    "best_msg": ("Highest estimated value: accept {rate:.0%} of loans (threshold {thr:.3f}, bad rate {bad:.1%}).", "Valeur estimée maximale : accepter {rate:.0%} des prêts (seuil {thr:.3f}, taux de mauvais prêts {bad:.1%}).", "أعلى قيمة مقدّرة: اقبل {rate:.0%} من القروض (العتبة {thr:.3f}، معدل القروض السيئة {bad:.1%})."),
    "acc_bad_title": ("Acceptance and bad rates", "Acceptation et mauvais prêts", "القبول ومعدل القروض السيئة"),
    "value_title": ("Estimated value by acceptance rate", "Valeur estimée selon le taux d'acceptation", "القيمة المقدّرة حسب معدل القبول"),
    "strategy_table": ("Strategy table", "Tableau de stratégie", "جدول الاستراتيجية"),
    "download_table": ("Download the strategy table (CSV)", "Télécharger le tableau (CSV)", "تنزيل جدول الاستراتيجية (CSV)"),
    "score_title": ("Score a new loan application", "Évaluer une nouvelle demande de prêt", "تقييم طلب قرض جديد"),
    "age": ("Age", "Âge", "العمر"),
    "income": ("Yearly income ($)", "Revenu annuel ($)", "الدخل السنوي ($)"),
    "emp": ("Employment length (years)", "Ancienneté professionnelle (ans)", "مدة العمل (سنوات)"),
    "home": ("Home ownership", "Logement", "ملكية السكن"),
    "purpose": ("Loan purpose", "Objet du prêt", "غرض القرض"),
    "grade": ("Loan grade", "Note du prêt", "درجة القرض"),
    "amount": ("Loan amount ($)", "Montant du prêt ($)", "مبلغ القرض ($)"),
    "rate": ("Interest rate (%)", "Taux d'intérêt (%)", "سعر الفائدة (%)"),
    "prev_default": ("Previous default on file", "Défaut antérieur au dossier", "تعثر سابق في السجل"),
    "hist": ("Credit history length (years)", "Historique de crédit (ans)", "مدة السجل الائتماني (سنوات)"),
    "score_btn": ("Score this loan", "Évaluer ce prêt", "قيّم هذا القرض"),
    "p_gbt": ("Default probability (GBT)", "Probabilité de défaut (arbre)", "احتمال التعثر (الشجرة)"),
    "p_lr": ("Default probability (logistic)", "Probabilité de défaut (logistique)", "احتمال التعثر (لوجستي)"),
    "exp_loss": ("Expected loss", "Perte attendue", "الخسارة المتوقعة"),
    "decision_at": ("Decision at {rate:.0%} acceptance", "Décision à {rate:.0%} d'acceptation", "القرار عند قبول {rate:.0%}"),
    "w_accept": ("Accept", "Accepter", "قبول"),
    "w_reject": ("Reject", "Refuser", "رفض"),
    "decision_banner": ("Decision: {word}", "Décision : {word}", "القرار: {word}"),
    "decision_sub": ("Default probability {p:.1%} vs acceptance threshold {thr:.1%}", "Probabilité de défaut {p:.1%} contre seuil d'acceptation {thr:.1%}", "احتمال التعثر {p:.1%} مقابل عتبة القبول {thr:.1%}"),
    "score_caption": ("The loan is accepted when its default probability is at or below {thr:.3f}. Change the acceptance rate and loss given default in the sidebar.", "Le prêt est accepté si sa probabilité de défaut est inférieure ou égale à {thr:.3f}. Modifiez le taux d'acceptation et la perte en cas de défaut dans la barre latérale.", "يُقبل القرض إذا كان احتمال تعثره أقل من {thr:.3f} أو يساويه. غيّر معدل القبول والخسارة عند التعثر من الشريط الجانبي."),
    "chat_title": ("AI Chatbot", "Chatbot IA", "شات بوت ذكي"),
    "chat_caption": ("Ask for an explanation of the data, the models, the metrics or the numbers on screen.", "Demandez une explication sur les données, les modèles, les indicateurs ou les chiffres affichés.", "اطلب شرحًا للبيانات أو النماذج أو المؤشرات أو الأرقام المعروضة."),
    "api_label": ("Anthropic API key (optional)", "Clé API Anthropic (facultative)", "مفتاح Anthropic API (اختياري)"),
    "api_help": ("Kept only for this session. Never share it or put it in the code.", "Conservée pour cette session uniquement. Ne la partagez jamais et ne l'écrivez pas dans le code.", "تُحفظ لهذه الجلسة فقط. لا تشاركها ولا تكتبها داخل الكود."),
    "lib_missing": ("The anthropic library is missing. In Terminal, run:  pip install anthropic", "La bibliothèque anthropic est absente. Dans le Terminal, lancez :  pip install anthropic", "مكتبة anthropic غير مثبتة. في الطرفية نفّذ:  pip install anthropic"),
    "offline_note": ("Built-in mode: answers come from the dashboard's own explanations, with no account or key needed. An optional API key unlocks free-form AI answers.", "Mode intégré : les réponses viennent des explications du tableau de bord, sans compte ni clé. Une clé API facultative permet des réponses IA libres.", "الوضع المدمج: الإجابات من شروحات اللوحة نفسها، دون حساب أو مفتاح. يتيح مفتاح API اختياري إجابات ذكية حرة."),
    "llm_note": ("AI mode: free-form answers generated by Claude.", "Mode IA : réponses libres générées par Claude.", "وضع الذكاء الاصطناعي: إجابات حرة يولّدها Claude."),
    "starter1": ("Explain this project", "Explique ce projet", "اشرح هذا المشروع"),
    "starter2": ("What do AUC and bad rate mean?", "Que signifient l'AUC et le taux de mauvais prêts ?", "ماذا يعني AUC ومعدل القروض السيئة؟"),
    "starter3": ("Explain my current results", "Explique mes résultats actuels", "اشرح نتائجي الحالية"),
    "clear": ("Clear chat", "Effacer la conversation", "مسح المحادثة"),
    "chat_ph": ("Ask about the data, models or results...", "Posez une question sur les données, modèles ou résultats...", "اسأل عن البيانات أو النماذج أو النتائج..."),
    "chat_err": ("The chatbot could not answer: {e}", "Le chatbot n'a pas pu répondre : {e}", "تعذّر على الشات بوت الإجابة: {e}"),
    "footer": ("Designed &amp; developed by", "Conçu &amp; développé par", "تصميم وتطوير"),
}
TR.update({
    "tab_analytics": ("Analytics", "Analyses", "التحليلات"),
    "confusion": ("Confusion matrices", "Matrices de confusion", "مصفوفات الالتباس"),
    "actual": ("Actual", "Réel", "الفعلي"), "predicted": ("Predicted", "Prédit", "المتوقع"),
    "cm_nd": ("Non-default", "Non-défaut", "غير متعثر"), "cm_d": ("Default", "Défaut", "متعثر"),
    "pr_title": ("Precision-recall curve", "Courbe précision-rappel", "منحنى الدقة والاسترجاع"),
    "precision": ("Precision", "Précision", "الدقة"), "recall": ("Recall", "Rappel", "الاسترجاع"),
    "imp_title": ("Feature importance (tree)", "Importance des variables (arbre)", "أهمية المتغيرات (الشجرة)"),
    "imp_x": ("Drop in AUC when shuffled", "Baisse d'AUC après permutation", "انخفاض AUC عند الخلط"),
    "port_el": ("Portfolio expected loss", "Perte attendue du portefeuille", "الخسارة المتوقعة للمحفظة"),
    "stress_title": ("Stress test", "Test de résistance", "اختبار الضغط"),
    "stress_slider": ("Default uplift (recession scenario)", "Hausse des défauts (scénario de récession)", "زيادة التعثر (سيناريو ركود)"),
    "base": ("Base", "Base", "الأساسي"), "stressed": ("Stressed", "Sous stress", "تحت الضغط"),
    "cost_title": ("Cost-based cutoff", "Seuil basé sur les coûts", "العتبة حسب التكلفة"),
    "cost_bad": ("Cost of accepting a bad loan ($)", "Coût d'un mauvais prêt accepté ($)", "تكلفة قبول قرض سيئ ($)"),
    "cost_good": ("Cost of rejecting a good loan ($)", "Coût d'un bon prêt refusé ($)", "تكلفة رفض قرض جيد ($)"),
    "best_cutoff": ("Best cutoff: {c:.2f} (total cost ${cost:,.0f})", "Meilleur seuil : {c:.2f} (coût total {cost:,.0f} $)", "أفضل عتبة: {c:.2f} (التكلفة الإجمالية {cost:,.0f} دولار)"),
    "cost_chart": ("Total cost by cutoff", "Coût total selon le seuil", "التكلفة الإجمالية حسب العتبة"),
    "cutoff_x": ("Cutoff", "Seuil", "العتبة"),
    "scen_title": ("Saved scenarios", "Scénarios enregistrés", "السيناريوهات المحفوظة"),
    "scen_save": ("Save this scenario", "Enregistrer ce scénario", "احفظ هذا السيناريو"),
    "scen_clear": ("Clear scenarios", "Effacer les scénarios", "مسح السيناريوهات"),
    "pdf": ("Download PDF report", "Télécharger le rapport PDF", "تنزيل تقرير PDF"),
    "gauge": ("Credit score", "Score de crédit", "درجة الائتمان"),
    "band_ex": ("Excellent", "Excellent", "ممتاز"), "band_good": ("Good", "Bon", "جيد"),
    "band_fair": ("Fair", "Moyen", "مقبول"), "band_poor": ("Poor", "Faible", "ضعيف"),
    "reasons": ("Main factors behind this result", "Principaux facteurs de ce résultat", "العوامل الرئيسية وراء هذه النتيجة"),
    "raises": ("raises risk", "augmente le risque", "يرفع المخاطر"), "lowers": ("lowers risk", "réduit le risque", "يخفض المخاطر"),
    "whatif": ("What-if analysis", "Analyse what-if", "تحليل ماذا لو"),
    "wi_income": ("Income change (%)", "Variation du revenu (%)", "تغيّر الدخل (%)"),
    "wi_amount": ("Loan amount change (%)", "Variation du montant (%)", "تغيّر مبلغ القرض (%)"),
    "wi_rate": ("Interest rate change (points)", "Variation du taux (points)", "تغيّر سعر الفائدة (نقاط)"),
    "wi_result": ("Default probability: {a:.1%} to {b:.1%} ({d:+.1f} points)", "Probabilité de défaut : {a:.1%} puis {b:.1%} ({d:+.1f} points)", "احتمال التعثر: {a:.1%} ثم {b:.1%} ({d:+.1f} نقطة)"),
    "batch_title": ("Batch scoring", "Évaluation par lot", "التقييم الدفعي"),
    "batch_help": ("Upload a CSV of applicants (same columns, loan_status not needed).", "Importez un CSV de demandeurs (mêmes colonnes, loan_status non requis).", "ارفع ملف CSV للمتقدمين (الأعمدة نفسها، ولا يلزم loan_status)."),
    "batch_dl": ("Download scored file", "Télécharger le fichier évalué", "تنزيل الملف المقيَّم"),
    "batch_done": ("{n:,} applicants scored: {acc:,} accepted, {rej:,} rejected", "{n:,} demandeurs évalués : {acc:,} acceptés, {rej:,} refusés", "تم تقييم {n:,} متقدم: {acc:,} مقبول و{rej:,} مرفوض"),
    "an_filters": ("Filters", "Filtres", "المرشحات"),
    "an_age": ("Age range", "Tranche d'âge", "الفئة العمرية"),
    "no_data": ("No loans match these filters.", "Aucun prêt ne correspond à ces filtres.", "لا توجد قروض تطابق هذه المرشحات."),
    "heat_title": ("Default rate by grade and purpose", "Taux de défaut par note et objet", "معدل التعثر حسب الدرجة والغرض"),
    "fair_title": ("Fairness check by age group", "Contrôle d'équité par tranche d'âge", "فحص الإنصاف حسب الفئة العمرية"),
    "age_group": ("Age group", "Tranche d'âge", "الفئة العمرية"),
    "approval": ("Approval rate", "Taux d'approbation", "معدل الموافقة"),
    "disparity": ("Approval ratio (lowest / highest): {r:.2f}", "Rapport d'approbation (min / max) : {r:.2f}", "نسبة الموافقة (الأدنى / الأعلى): {r:.2f}"),
    "fair_note": ("A ratio under 0.80 may signal unequal treatment and deserves review.", "Un rapport inférieur à 0,80 peut signaler un traitement inégal et mérite un examen.", "نسبة أقل من 0.80 قد تشير إلى معاملة غير متكافئة وتستحق المراجعة."),
    "dq_title": ("Data quality", "Qualité des données", "جودة البيانات"),
    "dq_missing": ("Missing values", "Valeurs manquantes", "القيم المفقودة"),
    "dq_removed": ("Rows removed by cleaning", "Lignes retirées par le nettoyage", "الصفوف المحذوفة بالتنظيف"),
    "dq_dup": ("Duplicate rows", "Lignes en double", "الصفوف المكررة"),
    "corr_title": ("Correlation heatmap", "Carte de corrélation", "خريطة الارتباط"),
})
FEAT = {"person_income": ("Income", "Revenu", "الدخل"), "loan_int_rate": ("Interest rate", "Taux d'intérêt", "سعر الفائدة"),
        "loan_percent_income": ("Loan-to-income ratio", "Ratio prêt/revenu", "نسبة القرض إلى الدخل"),
        "loan_amnt": ("Loan amount", "Montant du prêt", "مبلغ القرض"), "person_age": ("Age", "Âge", "العمر"),
        "person_emp_length": ("Employment length", "Ancienneté pro.", "مدة العمل"),
        "cb_person_cred_hist_length": ("Credit history", "Historique de crédit", "السجل الائتماني"),
        "loan_status": ("Default", "Défaut", "التعثر")}
PREFIX = {"loan_grade_": ("Grade {v}", "Note {v}", "الدرجة {v}"), "loan_intent_": ("Purpose: {v}", "Objet : {v}", "الغرض: {v}"),
          "person_home_ownership_": ("Housing: {v}", "Logement : {v}", "السكن: {v}"),
          "cb_person_default_on_file_": ("Previous default: {v}", "Défaut antérieur : {v}", "تعثر سابق: {v}")}
PV = {"DEBTCONSOLIDATION": ("Debt consolidation", "Regroupement de dettes", "توحيد الديون"),
      "HOMEIMPROVEMENT": ("Home improvement", "Rénovation de l'habitat", "تحسين المنزل"),
      "EDUCATION": ("Education", "Éducation", "التعليم"), "MEDICAL": ("Medical", "Médical", "طبي"),
      "PERSONAL": ("Personal", "Personnel", "شخصي"), "VENTURE": ("Venture", "Création d'entreprise", "مشروع تجاري"),
      "RENT": ("Rent", "Location", "إيجار"), "OWN": ("Own", "Propriétaire", "ملك"),
      "MORTGAGE": ("Mortgage", "Crédit immobilier", "رهن عقاري"), "OTHER": ("Other", "Autre", "أخرى"),
      "N": ("No", "Non", "لا"), "Y": ("Yes", "Oui", "نعم"), "All": ("All", "Total", "الإجمالي")}
LI = 0  # language index: 0 = English, 1 = Français, 2 = العربية (set from the sidebar)


def t(key, **kw):
    text = TR[key][LI]
    return text.format(**kw) if kw else text


def tc(key):
    """Text for charts. Arabic needs reshaping; falls back to English if the helper libraries are missing."""
    if LI == 2:
        try:
            from arabic_reshaper import reshape
            from bidi.algorithm import get_display
            return get_display(reshape(TR[key][2]))
        except ImportError:
            return TR[key][0]
    return TR[key][LI]


def pretty(value):
    """Readable, translated text for display only (the models still use the original values)."""
    return PV[value][LI] if value in PV else str(value).title()


def shape(text, fallback=None):
    """Arabic text needs reshaping to draw correctly inside charts."""
    if LI == 2:
        try:
            from arabic_reshaper import reshape
            from bidi.algorithm import get_display
            return get_display(reshape(text))
        except ImportError:
            return fallback if fallback is not None else text
    return text


def feat_label(col):
    if col in FEAT:
        return FEAT[col][LI]
    for prefix, template in PREFIX.items():
        if col.startswith(prefix):
            return template[LI].format(v=pretty(col[len(prefix):]))
    return col


ACCEPT_RATES = np.round(np.linspace(1.0, 0.05, 20), 2)  # 1.00, 0.95, ..., 0.05


# ----------------------------------------------------------------------------
# 1. Helper functions
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Cleaning the data...")
def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Same cleaning steps as the analysis script (chapter 1)."""
    df = raw.copy()
    df = df[~(df["person_emp_length"] > 60)]          # impossible employment lengths
    df = df[~(df["person_age"] > 100)]                # impossible ages
    df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())
    df = df.dropna(subset=["loan_int_rate"])          # drop rows with no interest rate
    return df.reset_index(drop=True)


def one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the text columns into 0/1 columns (same as chapter 2 of the script)."""
    numbers = df.select_dtypes(include="number")
    text = df.select_dtypes(exclude="number")
    return pd.concat([numbers, pd.get_dummies(text, dtype=int)], axis=1)


@st.cache_resource(show_spinner="Training the two models (only happens once)...")
def train_models(df: pd.DataFrame) -> dict:
    """Train the logistic regression and the gradient boosted tree."""
    prepared = one_hot(df)
    X = prepared.drop("loan_status", axis=1)
    y = prepared["loan_status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)

    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    gbt = HistGradientBoostingClassifier(random_state=123).fit(X_train, y_train)

    return {
        "columns": list(X.columns),
        "y_test": y_test.to_numpy(),
        "loan_amnt_test": X_test["loan_amnt"].to_numpy(),
        "X_test": X_test,
        "lr_probs": logistic.predict_proba(X_test)[:, 1],
        "gbt_probs": gbt.predict_proba(X_test)[:, 1],
        "logistic": logistic,
        "gbt": gbt,
    }


def strategy_row(true_status, probs, loan_amnt, rate):
    """Numbers for ONE acceptance rate: threshold, bad rate, estimated value."""
    threshold = np.quantile(probs, rate).round(3)     # rounded to 3 decimals, like the script
    accepted = probs <= threshold                     # we accept the safest `rate` share
    bad_rate = (true_status[accepted].sum() / accepted.sum()).round(3)
    num_accepted = int(rate * len(probs))
    avg_loan = loan_amnt.mean()
    value = (num_accepted * (1 - bad_rate) * avg_loan) - (num_accepted * bad_rate * avg_loan)
    return {"Acceptance Rate": rate, "Threshold": threshold, "Bad Rate": bad_rate,
            "Num Accepted Loans": num_accepted, "Avg Loan Amnt": avg_loan,
            "Estimated Value": value}


def build_strategy_table(true_status, probs, loan_amnt) -> pd.DataFrame:
    return pd.DataFrame([strategy_row(true_status, probs, loan_amnt, r) for r in ACCEPT_RATES])


# ----------------------------------------------------------------------------
# 2. Sidebar: data + controls
# ----------------------------------------------------------------------------
_LANGS = ["English", "Français", "العربية"]
_choice = st.sidebar.selectbox("Language / Langue / اللغة", _LANGS)
LI = _LANGS.index(_choice)
if LI == 2:
    st.markdown("""<style>
[data-testid="stMain"], section.main, [data-testid="stSidebarContent"] { direction: rtl; text-align: right; }
[data-testid="stSlider"], [data-testid="stNumberInput"] input, [data-testid="stDataFrame"], [data-testid="stCode"], .stPyplot { direction: ltr; }
.stApp .hero .sig { right: auto; left: 34px; }
</style>""", unsafe_allow_html=True)
st.sidebar.title(t("controls"))

source = st.sidebar.radio(t("data_source"), ["orig", "up"], format_func=lambda k: t("src_" + k))
uploaded = None
if source == "up":
    uploaded = st.sidebar.file_uploader(t("csv_label"), type="csv",
                                        help=t("upload_help"))
if uploaded is not None:
    raw = pd.read_csv(uploaded)
    data_label = t("lbl_up", name=uploaded.name)
elif DEFAULT_CSV.exists():
    raw = pd.read_csv(DEFAULT_CSV)
    data_label = t("lbl_orig")
else:
    st.title("Credit Risk Dashboard")
    st.info("Upload credit_risk_dataset.csv in the sidebar, or copy it into the same folder as app.py "
            "and refresh the page.")
    st.stop()

missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
if missing:
    st.error(t("missing", cols=', '.join(missing)))
    st.stop()

st.sidebar.caption(t("active", label=data_label, n=len(raw)))
if st.sidebar.button(t("reset")):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

fx_on = st.sidebar.toggle(t("effects"), value=True)
cutoff = st.sidebar.slider(t("cutoff"), 0.10, 0.90, 0.40, 0.05,
                           help=t("cutoff_help"))
accept_rate = st.sidebar.slider(t("accept"), 0.05, 1.00, 0.85, 0.05,
                                help=t("accept_help"))
lgd = st.sidebar.slider(t("lgd"), 0.0, 1.0, 0.20, 0.05,
                        help=t("lgd_help"))

df = clean_data(raw)
model = train_models(df)
y_test = model["y_test"]
lr_probs = model["lr_probs"]
gbt_probs = model["gbt_probs"]

st.markdown(f"""
<div class="hero"><span class="pill">{t('pill')}</span>
<h1>{t('title')}</h1>
<p>{t('subtitle')}</p></div>
""", unsafe_allow_html=True)

tab_data, tab_models, tab_strategy, tab_score, tab_analytics, tab_chat = st.tabs(
    [t("tab_data"), t("tab_models"), t("tab_strategy"), t("tab_score"), t("tab_analytics"), t("tab_chat")])

# ----------------------------------------------------------------------------
# 3a. Data tab
# ----------------------------------------------------------------------------
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric(t("loans"), f"{len(df):,}")
    c2.metric(t("default_rate"), f"{df['loan_status'].mean():.1%}")
    c3.metric(t("avg_loan"), f"${df['loan_amnt'].mean():,.0f}")

    st.subheader(t("first_rows"))
    st.dataframe(df.head(10), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader(t("loan_amounts"))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["loan_amnt"], bins="auto", color=TEAL, alpha=0.9)
        ax.set_xlabel(tc("loan_amount_x"))
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader(t("intent_status"))
        st.dataframe(pd.crosstab(df["loan_intent"], df["loan_status"], margins=True).rename(columns=str, index=pretty),
                     width="stretch")

# ----------------------------------------------------------------------------
# 3b. Model comparison tab
# ----------------------------------------------------------------------------
with tab_models:
    lr_labels = (lr_probs > cutoff).astype(int)
    gbt_labels = (gbt_probs > cutoff).astype(int)
    names = ["Non-Default", "Default"]

    st.subheader(t("reports_at", cutoff=cutoff))
    left, right = st.columns(2)
    for col, title, labels, probs in [(left, t("lr"), lr_labels, lr_probs),
                                      (right, t("gbt"), gbt_labels, gbt_probs)]:
        with col:
            st.markdown(f"**{title}**")
            st.code(classification_report(y_test, labels, target_names=names))
            f1 = precision_recall_fscore_support(y_test, labels, average="macro")[2]
            m1, m2 = st.columns(2)
            m1.metric("Macro F1", f"{f1:.2f}")
            m2.metric("AUC", f"{roc_auc_score(y_test, probs):.2f}")

    st.subheader(t("roc_cal"))
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for probs, label, color in [(lr_probs, "Logistic Regression", NAVY),
                                    (gbt_probs, "GBT", TEAL)]:
            fallout, sensitivity, _ = roc_curve(y_test, probs)
            ax.plot(fallout, sensitivity, color=color, label=label)
        ax.plot([0, 1], [0, 1], linestyle="--", label="Random Prediction")
        ax.set_title(tc("roc_title"))
        ax.set_xlabel(tc("fallout"))
        ax.set_ylabel(tc("sensitivity"))
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        for probs, label in [(lr_probs, "Logistic Regression"), (gbt_probs, "Gradient Boosted tree")]:
            frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
            ax.plot(mean_pred, frac_pos, "s-", label=label)
        ax.set_title(tc("cal_title"))
        ax.set_xlabel(tc("avg_pred"))
        ax.set_ylabel(tc("frac_pos"))
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

# ----------------------------------------------------------------------------
# 3c. Acceptance strategy tab (uses the gradient boosted tree)
# ----------------------------------------------------------------------------
with tab_strategy:
    strat_df = build_strategy_table(y_test, gbt_probs, model["loan_amnt_test"])
    chosen = strategy_row(y_test, gbt_probs, model["loan_amnt_test"], accept_rate)

    accepted_mask = gbt_probs <= chosen["Threshold"]
    expected_loss = np.sum(gbt_probs[accepted_mask] * lgd * model["loan_amnt_test"][accepted_mask])

    st.subheader(t("if_accept", rate=accept_rate))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("threshold"), f"{chosen['Threshold']:.3f}")
    c2.metric(t("bad_rate"), f"{chosen['Bad Rate']:.1%}")
    c3.metric(t("est_value"), f"${chosen['Estimated Value']:,.0f}")
    c4.metric(t("exp_loss_acc"), f"${expected_loss:,.0f}")

    best = strat_df.loc[strat_df["Estimated Value"].idxmax()]
    st.success(t("best_msg", rate=best['Acceptance Rate'], thr=best['Threshold'], bad=best['Bad Rate']))

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(strat_df["Acceptance Rate"], strat_df["Bad Rate"], color=RED, lw=2.5)
        ax.axvline(accept_rate, color=GOLD, linestyle="--", lw=2)
        ax.set_title(tc("acc_bad_title"))
        ax.set_xlabel(tc("accept"))
        ax.set_ylabel(tc("bad_rate"))
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(strat_df["Acceptance Rate"], strat_df["Estimated Value"], color=TEAL, lw=2.5)
        ax.axvline(accept_rate, color=GOLD, linestyle="--", lw=2)
        ax.set_title(tc("value_title"))
        ax.set_xlabel(tc("accept"))
        ax.set_ylabel(tc("est_value"))
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader(t("strategy_table"))
    st.dataframe(strat_df.style.format({
        "Acceptance Rate": "{:.2f}", "Threshold": "{:.3f}", "Bad Rate": "{:.3f}",
        "Avg Loan Amnt": "${:,.0f}", "Estimated Value": "${:,.0f}"}),
        width="stretch")
    st.download_button(t("download_table"), strat_df.to_csv(index=False),
                       file_name="strategy_table.csv", mime="text/csv")

# ----------------------------------------------------------------------------
# 3d. Score a single loan
# ----------------------------------------------------------------------------
with tab_score:
    st.subheader(t("score_title"))
    with st.form("loan_form"):
        a, b, c = st.columns(3)
        person_age = a.number_input(t("age"), 18, 100, 30)
        person_income = a.number_input(t("income"), 1000, 10_000_000, 60_000, step=1000)
        person_emp_length = a.number_input(t("emp"), 0, 60, 5)
        person_home_ownership = b.selectbox(t("home"), sorted(df["person_home_ownership"].unique()), format_func=pretty)
        loan_intent = b.selectbox(t("purpose"), sorted(df["loan_intent"].unique()), format_func=pretty)
        loan_grade = b.selectbox(t("grade"), sorted(df["loan_grade"].unique()))
        loan_amnt = c.number_input(t("amount"), 500, 1_000_000, 10_000, step=500)
        loan_int_rate = c.number_input(t("rate"), 0.0, 40.0, 11.0, step=0.1)
        cb_default = c.selectbox(t("prev_default"), sorted(df["cb_person_default_on_file"].unique()), format_func=pretty)
        cb_hist = c.number_input(t("hist"), 0, 60, 4)
        submitted = st.form_submit_button(t("score_btn"))

    if submitted or st.session_state.get("scored"):
        st.session_state["scored"] = True
        applicant = pd.DataFrame([{
            "person_age": person_age, "person_income": person_income,
            "person_home_ownership": person_home_ownership, "person_emp_length": person_emp_length,
            "loan_intent": loan_intent, "loan_grade": loan_grade, "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate, "loan_percent_income": loan_amnt / person_income,
            "cb_person_default_on_file": cb_default, "cb_person_cred_hist_length": cb_hist,
        }])
        # Same one-hot columns as the training data (missing ones become 0)
        row = one_hot(applicant).reindex(columns=model["columns"], fill_value=0)

        pd_gbt = float(model["gbt"].predict_proba(row)[0, 1])
        pd_lr = float(model["logistic"].predict_proba(row)[0, 1])
        threshold = chosen["Threshold"]
        decision = "Accept" if pd_gbt <= threshold else "Reject"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(t("p_gbt"), f"{pd_gbt:.1%}")
        c2.metric(t("p_lr"), f"{pd_lr:.1%}")
        c3.metric(t("exp_loss"), f"${pd_gbt * lgd * loan_amnt:,.0f}")
        c4.metric(t("decision_at", rate=accept_rate), t("w_" + decision.lower()))
        css_class = "ok" if decision == "Accept" else "no"
        st.markdown(f'<div class="decision {css_class}">{t("decision_banner", word=t("w_" + decision.lower()))}'
                    f'<small>{t("decision_sub", p=pd_gbt, thr=threshold)}</small></div>',
                    unsafe_allow_html=True)
        st.caption(t("score_caption", thr=threshold))


# ----------------------------------------------------------------------------
# 3f. Extra features
# ----------------------------------------------------------------------------
def make_pdf():
    """One-click PDF report (English labels) built with matplotlib only."""
    from matplotlib.backends.backend_pdf import PdfPages
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        lines = [f"Data: {data_label}", f"Loans after cleaning: {len(df):,}", f"Default rate: {df['loan_status'].mean():.1%}",
                 f"Average loan: ${df['loan_amnt'].mean():,.0f}",
                 f"AUC - logistic regression: {roc_auc_score(y_test, lr_probs):.3f} | tree: {roc_auc_score(y_test, gbt_probs):.3f}",
                 "", f"Acceptance rate: {accept_rate:.0%} | loss given default: {lgd:.0%} | cutoff: {cutoff:.2f}",
                 f"Threshold: {chosen['Threshold']:.3f} | bad rate: {chosen['Bad Rate']:.1%}",
                 f"Estimated value: ${chosen['Estimated Value']:,.0f} | expected loss (accepted): ${expected_loss:,.0f}",
                 f"Best acceptance rate: {best['Acceptance Rate']:.0%} (value ${best['Estimated Value']:,.0f})"]
        fig.text(.08, .94, "Credit Risk Report", fontsize=22, weight="bold", color=NAVY)
        for i, line in enumerate(lines):
            fig.text(.08, .88 - i * .035, line, fontsize=11, color=NAVY)
        fig.text(.08, .04, "Mohamed AOUEL - Credit Risk Dashboard", fontsize=9, color=T["muted"])
        pdf.savefig(fig); plt.close(fig)
        fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69), facecolor="white")
        axes[0].plot(strat_df["Acceptance Rate"], strat_df["Bad Rate"], color=RED, lw=2.5); axes[0].set_title("Acceptance and bad rates")
        axes[1].plot(strat_df["Acceptance Rate"], strat_df["Estimated Value"], color=TEAL, lw=2.5); axes[1].set_title("Estimated value by acceptance rate")
        for ax in axes:
            ax.set_xlabel("Acceptance rate"); ax.axvline(accept_rate, color=GOLD, ls="--")
        pdf.savefig(fig); plt.close(fig)
    return buf.getvalue()


@st.cache_resource(show_spinner=False)
def tree_importance(_gbt, X, y, key):
    idx = np.random.default_rng(123).choice(len(y), size=min(3000, len(y)), replace=False)
    res = permutation_importance(_gbt, X.iloc[idx], y[idx], n_repeats=3, random_state=123, scoring="roc_auc")
    return pd.Series(res.importances_mean, index=X.columns).sort_values(ascending=False)


with tab_models:
    st.subheader(t("confusion"))
    left, right = st.columns(2)
    for col, name, probs in [(left, "lr", lr_probs), (right, "gbt", gbt_probs)]:
        cm = confusion_matrix(y_test, (probs > cutoff).astype(int))
        fig, ax = plt.subplots(figsize=(5, 3.8))
        ax.imshow(cm, cmap="Blues"); ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=12, weight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else NAVY)
        ax.set_xticks([0, 1]); ax.set_xticklabels([tc("cm_nd"), tc("cm_d")]); ax.set_yticks([0, 1]); ax.set_yticklabels([tc("cm_nd"), tc("cm_d")])
        ax.set_xlabel(tc("predicted")); ax.set_ylabel(tc("actual")); ax.set_title(tc(name))
        with col:
            st.pyplot(fig)
        plt.close(fig)

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for probs, name in [(lr_probs, "lr"), (gbt_probs, "gbt")]:
            prec, rec, _ = precision_recall_curve(y_test, probs)
            ax.plot(rec, prec, label=tc(name), lw=2)
        ax.set_title(tc("pr_title")); ax.set_xlabel(tc("recall")); ax.set_ylabel(tc("precision")); ax.legend()
        st.pyplot(fig); plt.close(fig)
    with right:
        imp = tree_importance(model["gbt"], model["X_test"], y_test, (len(y_test), round(float(gbt_probs.mean()), 6)))
        top = imp.head(10)[::-1]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.barh([shape(feat_label(c), c) for c in top.index], top.values, color=TEAL)
        ax.set_title(tc("imp_title")); ax.set_xlabel(tc("imp_x"))
        st.pyplot(fig); plt.close(fig)

with tab_strategy:
    el_total = float(np.sum(gbt_probs * lgd * model["loan_amnt_test"]))
    st.metric(t("port_el"), f"${el_total:,.0f}")

    st.subheader(t("stress_title"))
    uplift = st.slider(t("stress_slider"), 0, 100, 30, 5, format="%d%%") / 100
    probs_s = np.clip(gbt_probs * (1 + uplift), 0, 1)
    avg_amt = model["loan_amnt_test"].mean()

    def stress_numbers(p):
        bad = float(p[accepted_mask].mean()); n_acc = int(accepted_mask.sum())
        return bad, n_acc * (1 - bad) * avg_amt - n_acc * bad * avg_amt, float(np.sum(p[accepted_mask] * lgd * model["loan_amnt_test"][accepted_mask]))

    b0, b1 = stress_numbers(gbt_probs), stress_numbers(probs_s)
    for label, base_v, stress_v, fmt, inv in [(t("bad_rate"), b0[0], b1[0], "{:.1%}", True), (t("est_value"), b0[1], b1[1], "${:,.0f}", False),
                                              (t("exp_loss"), b0[2], b1[2], "${:,.0f}", True)]:
        a, b, c = st.columns(3)
        a.metric(f"{label} - {t('base')}", fmt.format(base_v))
        b.metric(f"{label} - {t('stressed')}", fmt.format(stress_v), delta=fmt.format(stress_v - base_v).replace("$-", "-$"),
                 delta_color="inverse" if inv else "normal")

    st.subheader(t("cost_title"))
    k1, k2 = st.columns(2)
    cost_bad = k1.number_input(t("cost_bad"), 0, 10_000_000, int(avg_amt * lgd), step=500)
    cost_good = k2.number_input(t("cost_good"), 0, 10_000_000, int(avg_amt * 0.1), step=100)
    cuts = np.round(np.arange(0.05, 0.96, 0.01), 2)
    totals = np.array([cost_bad * np.sum((gbt_probs <= c) & (y_test == 1)) + cost_good * np.sum((gbt_probs > c) & (y_test == 0)) for c in cuts])
    best_i = int(totals.argmin())
    st.success(t("best_cutoff", c=cuts[best_i], cost=totals[best_i]))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(cuts, totals, color=NAVY, lw=2.5); ax.scatter([cuts[best_i]], [totals[best_i]], color=GOLD, s=90, zorder=3)
    ax.set_title(tc("cost_chart")); ax.set_xlabel(tc("cutoff_x"))
    st.pyplot(fig); plt.close(fig)

    st.subheader(t("scen_title"))
    st.session_state.setdefault("scen", [])
    s1, s2 = st.columns(2)
    if s1.button(t("scen_save")):
        st.session_state["scen"].append({t("accept"): accept_rate, t("lgd"): lgd, t("threshold"): chosen["Threshold"],
                                         t("bad_rate"): chosen["Bad Rate"], t("est_value"): chosen["Estimated Value"], t("exp_loss_acc"): expected_loss})
    if s2.button(t("scen_clear")):
        st.session_state["scen"] = []
    if st.session_state["scen"]:
        st.dataframe(pd.DataFrame(st.session_state["scen"]), width="stretch")
    try:
        st.download_button(t("pdf"), make_pdf(), file_name="credit_risk_report.pdf", mime="application/pdf")
    except Exception as e:
        st.warning(str(e))

with tab_score:
    if st.session_state.get("scored") and "pd_gbt" in globals():
        score = int(round(850 - pd_gbt * 550))
        band = t("band_ex") if score >= 740 else t("band_good") if score >= 670 else t("band_fair") if score >= 580 else t("band_poor")
        st.subheader(t("gauge"))
        st.markdown(f'<div class="gauge-score">{score} <span style="font-size:1rem;font-weight:600">/ 850 - {band}</span></div>'
                    f'<div class="gauge-bar"><div class="gauge-marker" style="left:{(score - 300) / 550 * 100:.1f}%"></div></div>',
                    unsafe_allow_html=True)

        pipe = model["logistic"]
        contrib = pipe[-1].coef_[0] * pipe[0].transform(row)[0]
        order = np.argsort(contrib)
        st.subheader(f"{t('reasons')} ({t('lr')})")
        up_col, down_col = st.columns(2)
        for box, idxs, word in [(up_col, [i for i in order[::-1][:3] if contrib[i] > 0], "raises"), (down_col, [i for i in order[:3] if contrib[i] < 0], "lowers")]:
            for i in idxs:
                box.markdown(f"**{feat_label(model['columns'][i])}**: {t(word)} ({contrib[i]:+.2f})")

        st.subheader(t("whatif"))
        w1, w2, w3 = st.columns(3)
        d_inc = w1.slider(t("wi_income"), -50, 100, 0, 5)
        d_amt = w2.slider(t("wi_amount"), -50, 100, 0, 5)
        d_rate = w3.slider(t("wi_rate"), -5.0, 5.0, 0.0, 0.5)
        alt = applicant.copy()
        alt["person_income"] = alt["person_income"] * (1 + d_inc / 100)
        alt["loan_amnt"] = alt["loan_amnt"] * (1 + d_amt / 100)
        alt["loan_int_rate"] = (alt["loan_int_rate"] + d_rate).clip(lower=0)
        alt["loan_percent_income"] = alt["loan_amnt"] / alt["person_income"]
        new_pd = float(model["gbt"].predict_proba(one_hot(alt).reindex(columns=model["columns"], fill_value=0))[0, 1])
        st.info(t("wi_result", a=pd_gbt, b=new_pd, d=(new_pd - pd_gbt) * 100))

    st.subheader(t("batch_title"))
    st.caption(t("batch_help"))
    batch = st.file_uploader("CSV", type="csv", key="batch_up")
    if batch is not None:
        try:
            b = pd.read_csv(batch)
            need = [c for c in REQUIRED_COLUMNS if c not in ("loan_status", "loan_percent_income")]
            absent = [c for c in need if c not in b.columns]
            if absent:
                st.error(t("missing", cols=", ".join(absent)))
            else:
                if "loan_percent_income" not in b.columns:
                    b["loan_percent_income"] = b["loan_amnt"] / b["person_income"]
                feats = b.drop(columns=[c for c in ["loan_status"] if c in b.columns]).copy()
                feats["person_emp_length"] = feats["person_emp_length"].fillna(df["person_emp_length"].median())
                feats["loan_int_rate"] = feats["loan_int_rate"].fillna(df["loan_int_rate"].median())
                Xb = one_hot(feats).reindex(columns=model["columns"], fill_value=0)
                b["default_probability_gbt"] = model["gbt"].predict_proba(Xb)[:, 1]
                b["default_probability_logistic"] = model["logistic"].predict_proba(Xb)[:, 1]
                b["decision"] = np.where(b["default_probability_gbt"] <= chosen["Threshold"], "Accept", "Reject")
                n_acc = int((b["decision"] == "Accept").sum())
                st.success(t("batch_done", n=len(b), acc=n_acc, rej=len(b) - n_acc))
                st.dataframe(b.head(20), width="stretch")
                st.download_button(t("batch_dl"), b.to_csv(index=False), file_name="scored_applicants.csv", mime="text/csv")
        except Exception as e:
            st.error(str(e))

with tab_analytics:
    st.subheader(t("an_filters"))
    f1, f2 = st.columns(2)
    lo, hi = int(df["person_age"].min()), int(df["person_age"].max())
    age_rng = f1.slider(t("an_age"), lo, hi, (lo, hi))
    intents = sorted(df["loan_intent"].unique())
    picked = f2.multiselect(t("purpose"), intents, default=intents, format_func=pretty)
    fdf = df[df["person_age"].between(*age_rng) & df["loan_intent"].isin(picked)]
    if fdf.empty:
        st.info(t("no_data"))
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric(t("loans"), f"{len(fdf):,}"); m2.metric(t("default_rate"), f"{fdf['loan_status'].mean():.1%}"); m3.metric(t("avg_loan"), f"${fdf['loan_amnt'].mean():,.0f}")
        st.subheader(t("heat_title"))
        pv = fdf.pivot_table(index="loan_grade", columns="loan_intent", values="loan_status", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(9, 4.6))
        im = ax.imshow(pv.values, cmap="RdYlGn_r", aspect="auto"); ax.grid(False)
        ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels([shape(pretty(c), str(c)) for c in pv.columns], rotation=30, ha="right")
        ax.set_yticks(range(len(pv.index))); ax.set_yticklabels(pv.index)
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                if not np.isnan(pv.values[i, j]):
                    ax.text(j, i, f"{pv.values[i, j]:.0%}", ha="center", va="center", fontsize=9, color=NAVY)
        fig.colorbar(im, ax=ax, fraction=.03)
        st.pyplot(fig); plt.close(fig)

    st.subheader(t("fair_title"))
    groups = pd.cut(model["X_test"]["person_age"].to_numpy(), [0, 24, 34, 44, 200], labels=["<25", "25-34", "35-44", "45+"])
    fair = pd.DataFrame({"g": groups, "acc": gbt_probs <= chosen["Threshold"], "y": y_test}).groupby("g", observed=True).agg(
        approval=("acc", "mean"), default=("y", "mean"), n=("y", "size"))
    left, right = st.columns(2)
    with left:
        show = fair.rename(columns={"approval": t("approval"), "default": t("default_rate"), "n": t("loans")})
        show.index.name = t("age_group")
        st.dataframe(show.style.format({t("approval"): "{:.1%}", t("default_rate"): "{:.1%}", t("loans"): "{:,}"}), width="stretch")
        ratio = float(fair["approval"].min() / fair["approval"].max())
        (st.success if ratio >= 0.8 else st.warning)(t("disparity", r=ratio))
        st.caption(t("fair_note"))
    with right:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.bar(fair.index.astype(str), fair["approval"], color=TEAL); ax.set_ylim(0, 1)
        ax.axhline(fair["approval"].max() * 0.8, color=GOLD, ls="--"); ax.set_title(tc("approval")); ax.set_xlabel(tc("age_group"))
        st.pyplot(fig); plt.close(fig)

    st.subheader(t("dq_title"))
    d1, d2, d3 = st.columns(3)
    d1.metric(t("dq_missing"), f"{int(raw.isnull().sum().sum()):,}"); d2.metric(t("dq_removed"), f"{len(raw) - len(df):,}"); d3.metric(t("dq_dup"), f"{int(raw.duplicated().sum()):,}")
    corr = df.select_dtypes(include="number").corr()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1); ax.grid(False)
    labels = [shape(feat_label(c), c) for c in corr.columns]
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_title(tc("corr_title")); fig.colorbar(im, ax=ax, fraction=.04)
    st.pyplot(fig); plt.close(fig)

# ----------------------------------------------------------------------------
# 3e. AI assistant tab
# ----------------------------------------------------------------------------
MODEL_NAME = "claude-haiku-5-5"   # fast and cheap; switch to "claude-sonnet-5-5" for deeper answers
STARTERS = [t("starter1"), t("starter2"), t("starter3")]

KB = {
    "overview": (
        "This dashboard predicts which loans are likely to default. The data is cleaned and split into training and test sets, then two models (logistic regression and a gradient boosted tree) estimate each loan's probability of default. You compare the models, choose how many loans to accept, and score a single applicant. Right now: {n:,} loans with a {default_rate:.1%} default rate.",
        "Ce tableau de bord prédit quels prêts risquent de faire défaut. Les données sont nettoyées et séparées en apprentissage et test, puis deux modèles (régression logistique et arbre à gradient boosté) estiment la probabilité de défaut de chaque prêt. Vous comparez les modèles, choisissez combien de prêts accepter et évaluez un demandeur. Actuellement : {n:,} prêts, taux de défaut de {default_rate:.1%}.",
        "تتنبأ هذه اللوحة بالقروض المرجّح تعثرها. تُنظَّف البيانات وتُقسَّم إلى تدريب واختبار، ثم يقدّر نموذجان (الانحدار اللوجستي وشجرة التعزيز التدريجي) احتمال تعثر كل قرض. بعدها تقارن النماذج وتحدد نسبة القروض التي تقبلها وتقيّم طلب قرض واحد. حاليًا: {n:,} قرض ومعدل تعثر {default_rate:.1%}."),
    "auc": (
        "AUC measures how well a model ranks risky loans above safe ones: 0.5 is random guessing and 1.0 is perfect. Here the logistic regression scores {auc_lr:.3f} and the tree {auc_gbt:.3f}.",
        "L'AUC mesure la capacité d'un modèle à classer les prêts risqués avant les prêts sûrs : 0,5 correspond au hasard et 1,0 à un modèle parfait. Ici, la régression logistique obtient {auc_lr:.3f} et l'arbre {auc_gbt:.3f}.",
        "يقيس AUC قدرة النموذج على ترتيب القروض الخطرة قبل الآمنة: القيمة 0.5 تعني تخمينًا عشوائيًا و1.0 تعني نموذجًا مثاليًا. هنا يحقق الانحدار اللوجستي {auc_lr:.3f} والشجرة {auc_gbt:.3f}."),
    "bad_rate": (
        "The bad rate is the share of accepted loans that actually defaulted. The lower it is, the healthier the accepted portfolio. It falls when you accept fewer loans, because only the safest ones are kept.",
        "Le taux de mauvais prêts est la part des prêts acceptés qui ont réellement fait défaut. Plus il est bas, plus le portefeuille accepté est sain. Il baisse quand vous acceptez moins de prêts, car seuls les plus sûrs sont conservés.",
        "معدل القروض السيئة هو نسبة القروض المقبولة التي تعثرت فعلًا. كلما انخفض كانت المحفظة المقبولة أسلم، وهو ينخفض عندما تقبل عددًا أقل من القروض لأنك تحتفظ بالأكثر أمانًا فقط."),
    "strategy": (
        "Accepting a share of loans means keeping the safest ones. The threshold is the default probability below which that share of loans falls. At the best acceptance rate ({best_rate:.0%}) the threshold is {best_thr:.3f}.",
        "Accepter une part des prêts revient à garder les plus sûrs. Le seuil est la probabilité de défaut sous laquelle se situe cette part des prêts. Au meilleur taux d'acceptation ({best_rate:.0%}), le seuil est de {best_thr:.3f}.",
        "قبول نسبة من القروض يعني الاحتفاظ بالأكثر أمانًا منها. العتبة هي الاحتمال الذي تقع تحته تلك النسبة من القروض. عند أفضل معدل قبول ({best_rate:.0%}) تكون العتبة {best_thr:.3f}."),
    "value": (
        "Estimated value = accepted loans x (1 - bad rate) x average loan - accepted loans x bad rate x average loan. It rewards repaid loans and penalises defaults. The best value is ${best_value:,.0f} at {best_rate:.0%} acceptance.",
        "Valeur estimée = prêts acceptés x (1 - taux de mauvais prêts) x prêt moyen - prêts acceptés x taux de mauvais prêts x prêt moyen. Elle récompense les prêts remboursés et pénalise les défauts. La meilleure valeur est de {best_value:,.0f} $ à {best_rate:.0%} d'acceptation.",
        "القيمة المقدّرة = القروض المقبولة × (1 − معدل القروض السيئة) × متوسط القرض − القروض المقبولة × معدل القروض السيئة × متوسط القرض. فهي تكافئ القروض المسدَّدة وتعاقب التعثر. أفضل قيمة هي {best_value:,.0f} دولار عند قبول {best_rate:.0%}."),
    "loss": (
        "Expected loss = probability of default x loss given default x loan amount. The loss given default slider (currently {lgd:.0%}) is the share of the amount you lose when a borrower defaults.",
        "Perte attendue = probabilité de défaut x perte en cas de défaut x montant du prêt. Le curseur de perte en cas de défaut (actuellement {lgd:.0%}) est la part du montant perdue quand l'emprunteur fait défaut.",
        "الخسارة المتوقعة = احتمال التعثر × الخسارة عند التعثر × مبلغ القرض. مؤشر الخسارة عند التعثر (حاليًا {lgd:.0%}) هو النسبة التي تخسرها من المبلغ عند تعثر المقترض."),
    "models": (
        "Logistic regression is a simple model that is easy to explain. The gradient boosted tree builds many small decision trees and can capture more complex patterns. Here the AUC is {auc_gbt:.3f} for the tree against {auc_lr:.3f} for the logistic regression.",
        "La régression logistique est un modèle simple et facile à expliquer. L'arbre à gradient boosté construit de nombreux petits arbres de décision et capte des relations plus complexes. Ici, l'AUC est de {auc_gbt:.3f} pour l'arbre contre {auc_lr:.3f} pour la régression logistique.",
        "الانحدار اللوجستي نموذج بسيط وسهل الشرح. أما شجرة التعزيز التدريجي فتبني عددًا كبيرًا من أشجار القرار الصغيرة وتلتقط علاقات أعقد. هنا قيمة AUC هي {auc_gbt:.3f} للشجرة مقابل {auc_lr:.3f} للانحدار اللوجستي."),
    "cleaning": (
        "Cleaning removes impossible values (employment length above 60 years, age above 100), fills missing employment length with the median, and drops loans with no interest rate. Text columns become 0/1 columns before training.",
        "Le nettoyage supprime les valeurs impossibles (ancienneté de plus de 60 ans, âge de plus de 100 ans), remplace l'ancienneté manquante par la médiane et retire les prêts sans taux d'intérêt. Les colonnes texte deviennent des colonnes 0/1 avant l'entraînement.",
        "يزيل التنظيف القيم المستحيلة (مدة عمل فوق 60 سنة أو عمر فوق 100)، ويملأ مدة العمل الناقصة بالوسيط، ويحذف القروض التي بلا سعر فائدة. ثم تتحول الأعمدة النصية إلى أعمدة 0/1 قبل التدريب."),
    "results": (
        "Current data: {n:,} loans, default rate {default_rate:.1%}, average loan ${avg_loan:,.0f}. AUC: logistic {auc_lr:.3f}, tree {auc_gbt:.3f}. With your sliders (cutoff {cutoff:.2f}, acceptance {accept_rate:.0%}, loss given default {lgd:.0%}) the best acceptance rate is {best_rate:.0%}, with a bad rate of {best_bad:.1%}.",
        "Données actuelles : {n:,} prêts, taux de défaut {default_rate:.1%}, prêt moyen {avg_loan:,.0f} $. AUC : logistique {auc_lr:.3f}, arbre {auc_gbt:.3f}. Avec vos curseurs (seuil {cutoff:.2f}, acceptation {accept_rate:.0%}, perte en cas de défaut {lgd:.0%}), le meilleur taux d'acceptation est de {best_rate:.0%}, avec un taux de mauvais prêts de {best_bad:.1%}.",
        "البيانات الحالية: {n:,} قرض، معدل التعثر {default_rate:.1%}، ومتوسط القرض {avg_loan:,.0f} دولار. قيمة AUC: {auc_lr:.3f} للوجستي و{auc_gbt:.3f} للشجرة. مع إعداداتك (العتبة {cutoff:.2f}، القبول {accept_rate:.0%}، الخسارة عند التعثر {lgd:.0%}) أفضل معدل قبول هو {best_rate:.0%} بمعدل قروض سيئة {best_bad:.1%}."),
    "score": (
        "The score a loan tab lets you enter one applicant's details. You get the default probability from both models, the expected loss, and an accept or reject decision based on the threshold of the acceptance rate you chose.",
        "L'onglet d'évaluation d'un prêt permet de saisir les informations d'un demandeur. Vous obtenez la probabilité de défaut des deux modèles, la perte attendue et une décision d'acceptation ou de refus selon le seuil du taux d'acceptation choisi.",
        "تتيح لك خانة تقييم قرض إدخال بيانات مقترض واحد. تحصل على احتمال التعثر من النموذجين والخسارة المتوقعة وقرار القبول أو الرفض بحسب عتبة معدل القبول الذي اخترته."),
}
KEYWORDS = {
    "overview": ["project", "projet", "dashboard", "tableau de bord", "مشروع", "اللوحة"],
    "auc": ["auc", "roc"],
    "bad_rate": ["bad rate", "mauvais", "السيئة"],
    "strategy": ["strategy", "stratégie", "strategie", "acceptance", "acceptation", "threshold", "seuil", "استراتيجية", "قبول", "عتبة"],
    "value": ["estimated value", "value", "valeur", "القيمة"],
    "loss": ["loss", "perte", "lgd", "الخسارة"],
    "models": ["model", "modèle", "modele", "logistic", "logistique", "tree", "arbre", "نموذج", "النموذج", "شجرة", "اللوجستي"],
    "cleaning": ["clean", "nettoy", "missing", "manquant", "outlier", "aberrant", "تنظيف"],
    "results": ["result", "résultat", "resultat", "current", "actuel", "نتائج", "نتيجة", "الحالية"],
    "score": ["score", "évaluer", "evaluer", "applicant", "demandeur", "تقييم"],
}
FAQ_FALLBACK = (
    "I can explain: the project, AUC, the bad rate, the acceptance strategy, the estimated value, the expected loss, the two models, data cleaning, your current results, and scoring a loan. Try one of the buttons above.",
    "Je peux expliquer : le projet, l'AUC, le taux de mauvais prêts, la stratégie d'acceptation, la valeur estimée, la perte attendue, les deux modèles, le nettoyage des données, vos résultats actuels et l'évaluation d'un prêt. Essayez l'un des boutons ci-dessus.",
    "يمكنني شرح: المشروع، AUC، معدل القروض السيئة، استراتيجية القبول، القيمة المقدّرة، الخسارة المتوقعة، النموذجين، تنظيف البيانات، نتائجك الحالية، وتقييم قرض. جرّب أحد الأزرار أعلاه.")


def faq_answer(question, ctx):
    """Built-in answers (no API needed): match keywords, return up to two explanations."""
    q = question.lower()
    hits = sorted(((sum(len(w) for w in words if w in q), topic) for topic, words in KEYWORDS.items()), reverse=True)
    topics = [topic for score, topic in hits if score][:2]
    if not topics:
        return FAQ_FALLBACK[LI]
    return "\\n\\n".join(KB[topic][LI].format(**ctx) for topic in topics)

with tab_chat:
    st.subheader(t("chat_title"))
    st.caption(t("chat_caption"))

    api_key = ""
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        pass
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        api_key = st.text_input(t("api_label"), type="password",
                                help=t("api_help"))

    use_llm = bool(api_key) and anthropic is not None
    if api_key and anthropic is None:
        st.error(t("lib_missing"))
    st.caption(t("llm_note") if use_llm else t("offline_note"))

    strat_chat = build_strategy_table(y_test, gbt_probs, model["loan_amnt_test"])
    best_row = strat_chat.loc[strat_chat["Estimated Value"].idxmax()]
    ctx = dict(n=len(df), default_rate=df["loan_status"].mean(), avg_loan=df["loan_amnt"].mean(),
               auc_lr=roc_auc_score(y_test, lr_probs), auc_gbt=roc_auc_score(y_test, gbt_probs),
               cutoff=cutoff, accept_rate=accept_rate, lgd=lgd, best_rate=best_row["Acceptance Rate"],
               best_thr=best_row["Threshold"], best_bad=best_row["Bad Rate"], best_value=best_row["Estimated Value"])
    system_prompt = f"""You are the AI chatbot of a credit risk dashboard built by Mohamed AOUEL.
Explain things clearly and simply, like a patient teacher, to someone learning credit risk.
Answer in {['English', 'French', 'Arabic'][LI]} (the dashboard language) unless the user writes in another language. Keep answers short unless asked for detail.
This is a learning project: never present it as real lending advice.

HOW THE PROJECT WORKS
- Cleaning: remove employment length above 60 years and ages above 100, fill missing employment length
  with the median, drop rows with no interest rate. Text columns are turned into 0/1 columns (one-hot).
- Data is split 60% train / 40% test. Two models predict the probability that a loan defaults
  (loan_status = 1): logistic regression and a gradient boosted tree.
- Classification report: uses the default cutoff slider (a loan is predicted as default above it).
- AUC: how well a model ranks risky loans above safe ones (0.5 = random, 1 = perfect).
- Acceptance strategy (uses the tree): accepting a share of loans means keeping the safest ones.
  Threshold = the probability below which that share of loans falls. Bad rate = share of accepted
  loans that actually defaulted. Estimated value = accepted * (1 - bad rate) * avg loan
  - accepted * bad rate * avg loan. Expected loss = probability * loss given default * loan amount.
- Tabs: Data, Model comparison, Acceptance strategy, Score a loan (one applicant), AI Chatbot.

CURRENT NUMBERS ON SCREEN
- Data: {data_label}; {len(df):,} loans after cleaning; default rate {df['loan_status'].mean():.1%};
  average loan ${df['loan_amnt'].mean():,.0f}.
- AUC: logistic regression {roc_auc_score(y_test, lr_probs):.3f}, gradient boosted tree {roc_auc_score(y_test, gbt_probs):.3f}.
- Sliders: default cutoff {cutoff:.2f}, acceptance rate {accept_rate:.0%}, loss given default {lgd:.0%}.
- Best acceptance rate by estimated value: {best_row['Acceptance Rate']:.0%}
  (threshold {best_row['Threshold']:.3f}, bad rate {best_row['Bad Rate']:.1%},
  estimated value ${best_row['Estimated Value']:,.0f})."""

    if "chat" not in st.session_state:
        st.session_state["chat"] = []

    starter = None
    cols = st.columns(len(STARTERS) + 1)
    for col, question in zip(cols, STARTERS):
        if col.button(question, key=f"starter_{question}"):
            starter = question
    if cols[-1].button(t("clear")):
        st.session_state["chat"] = []
        st.rerun()

    for msg in st.session_state["chat"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input(t("chat_ph")) or starter
    if prompt:
        st.session_state["chat"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            if use_llm:
                try:
                    client = anthropic.Anthropic(api_key=api_key)
                    with client.messages.stream(model=MODEL_NAME, max_tokens=1000, system=system_prompt,
                                                messages=st.session_state["chat"][-11:]) as stream:
                        reply = st.write_stream(stream.text_stream)
                except Exception as e:
                    st.warning(t("chat_err", e=e))
                    reply = faq_answer(prompt, ctx)
                    st.markdown(reply)
            else:
                reply = faq_answer(prompt, ctx)
                st.markdown(reply)
            st.session_state["chat"].append({"role": "assistant", "content": reply})

st.markdown(f'<div class="footer">{t("footer")} <span class="sigtxt">Mohamed AOUEL</span>'
            f' &middot; {t("title")}</div>', unsafe_allow_html=True)


FX_JS = r"""
<script>
(function () {
  var P = window.parent;
  P.__fxOff = __FXOFF__;
  if (P.document.getElementById('fx-boot')) return;
  function effects() {
    var d = document, W = window;
    var on = function () { return !W.__fxOff; };
    function meta(n, c) { var m = d.createElement('meta'); m.name = n; m.content = c; d.head.appendChild(m); }
    meta('theme-color', '#0B2545'); meta('mobile-web-app-capable', 'yes');
    meta('apple-mobile-web-app-capable', 'yes'); meta('apple-mobile-web-app-title', 'Credit Risk');
    var lm = d.createElement('link'); lm.rel = 'manifest'; lm.href = '/app/static/manifest.json'; d.head.appendChild(lm);
    var li = d.createElement('link'); li.rel = 'apple-touch-icon'; li.href = '/app/static/icon-192.png'; d.head.appendChild(li);
    // cursor glow, card spotlight and 3D tilt
    var glow = d.createElement('div'); glow.id = 'fx-glow'; d.body.appendChild(glow);
    var mouse = { x: -999, y: -999 }, tilted = null;
    function untilt(el) { if (el) { el.style.transform = ''; } }
    d.addEventListener('mousemove', function (e) {
      mouse.x = e.clientX; mouse.y = e.clientY;
      glow.style.display = on() ? 'block' : 'none';
      glow.style.transform = 'translate(' + (e.clientX - 150) + 'px,' + (e.clientY - 150) + 'px)';
      var t = e.target.closest ? e.target.closest('[data-testid="stMetric"], .stPyplot') : null;
      if (tilted && tilted !== t) { untilt(tilted); tilted = null; }
      if (!t) return;
      var r = t.getBoundingClientRect(), x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
      t.style.setProperty('--mx', (e.clientX - r.left) + 'px'); t.style.setProperty('--my', (e.clientY - r.top) + 'px');
      if (on() && t.matches('[data-testid="stMetric"]')) {
        t.style.transform = 'perspective(800px) rotateX(' + (-(y - .5) * 12) + 'deg) rotateY(' + ((x - .5) * 12) + 'deg) translateY(-6px)';
        tilted = t;
      }
    });
    // scroll progress bar
    var bar = d.createElement('div'); bar.id = 'fx-progress'; d.body.appendChild(bar);
    d.addEventListener('scroll', function (e) {
      var el = e.target === d ? d.documentElement : e.target;
      if (!el || !el.scrollHeight || el.scrollHeight <= el.clientHeight + 2) return;
      bar.style.width = (el.scrollTop / (el.scrollHeight - el.clientHeight) * 100) + '%';
    }, true);
    // button ripple
    d.addEventListener('click', function (e) {
      if (!on()) return;
      var b = e.target.closest ? e.target.closest('button') : null; if (!b) return;
      var r = b.getBoundingClientRect(), n = Math.max(r.width, r.height), s = d.createElement('span');
      s.className = 'fx-ripple';
      s.style.cssText = 'width:' + n + 'px;height:' + n + 'px;left:' + (e.clientX - r.left - n / 2) + 'px;top:' + (e.clientY - r.top - n / 2) + 'px';
      b.style.position = 'relative'; b.style.overflow = 'hidden'; b.appendChild(s);
      setTimeout(function () { s.remove(); }, 650);
    }, true);
    // count-up numbers on the metric cards
    var cu = new WeakMap(), sched = false;
    function tnode(el) { var w = d.createTreeWalker(el, NodeFilter.SHOW_TEXT), n; while ((n = w.nextNode())) { if (n.nodeValue.trim()) return n; } return null; }
    function animate(node) {
      var st = cu.get(node) || {}; cu.set(node, st); var txt = node.nodeValue;
      if (st.running) { if (txt !== st.last) st.stale = true; return; }
      if (st.done === txt) return;
      var m = txt.trim().match(/^([^\d-]*)(-?[\d,]*\.?\d+)(.*)$/);
      if (!m || !on()) { st.done = txt; return; }
      var target = parseFloat(m[2].replace(/,/g, '')), dec = (m[2].split('.')[1] || '').length, comma = m[2].indexOf(',') > -1, t0 = null;
      st.running = true; st.stale = false;
      function fmt(v) { var q = v.toFixed(dec); if (comma) { var p = q.split('.'); p[0] = p[0].replace(/\B(?=(\d{3})+(?!\d))/g, ','); q = p.join('.'); } return m[1] + q + m[3]; }
      function step(ts) {
        if (!t0) t0 = ts; var p = Math.min((ts - t0) / 1000, 1), e = 1 - Math.pow(1 - p, 3);
        if (p < 1 && !st.stale) { st.last = fmt(target * e); node.nodeValue = st.last; requestAnimationFrame(step); }
        else { st.running = false; if (st.stale) { st.done = null; scan(); return; } node.nodeValue = txt; st.done = txt; }
      }
      requestAnimationFrame(step);
    }
    function scan() { d.querySelectorAll('[data-testid="stMetricValue"]').forEach(function (el) { var n = tnode(el); if (n) animate(n); }); }
    new MutationObserver(function () { if (!sched) { sched = true; requestAnimationFrame(function () { sched = false; scan(); }); } })
      .observe(d.body, { childList: true, subtree: true, characterData: true });
    scan();
    // particle network
    if (W.matchMedia && W.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    if ('ontouchstart' in W || W.innerWidth < 768) return;
    var cv = d.createElement('canvas'); cv.id = 'fx-canvas'; d.body.appendChild(cv);
    var ctx = cv.getContext('2d'), w, h, pts = [];
    function size() { w = cv.width = W.innerWidth; h = cv.height = W.innerHeight; }
    size(); W.addEventListener('resize', size);
    for (var i = 0; i < 48; i++) pts.push({ x: Math.random() * W.innerWidth, y: Math.random() * W.innerHeight, vx: (Math.random() - .5) * .45, vy: (Math.random() - .5) * .45 });
    (function draw() {
      cv.style.display = on() ? 'block' : 'none';
      if (on() && !d.hidden) {
        ctx.clearRect(0, 0, w, h);
        for (var i = 0; i < pts.length; i++) {
          var a = pts[i]; a.x += a.vx; a.y += a.vy;
          if (a.x < 0 || a.x > w) a.vx *= -1; if (a.y < 0 || a.y > h) a.vy *= -1;
          ctx.beginPath(); ctx.arc(a.x, a.y, 2, 0, 6.283); ctx.fillStyle = 'rgba(0,168,143,.7)'; ctx.fill();
          for (var j = i + 1; j < pts.length; j++) {
            var b = pts[j], dx = a.x - b.x, dy = a.y - b.y, dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 130) { ctx.strokeStyle = 'rgba(19,66,122,' + (0.35 * (1 - dist / 130)) + ')'; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); }
          }
          var mx = a.x - mouse.x, my = a.y - mouse.y, md = Math.sqrt(mx * mx + my * my);
          if (md < 170) { ctx.strokeStyle = 'rgba(0,168,143,' + (0.55 * (1 - md / 170)) + ')'; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(mouse.x, mouse.y); ctx.stroke(); }
        }
      }
      W.requestAnimationFrame(draw);
    })();
  }
  var s = P.document.createElement('script');
  s.id = 'fx-boot';
  s.textContent = '(' + effects.toString() + ')();';
  P.document.head.appendChild(s);
})();
</script>
"""
components.html(FX_JS.replace("__FXOFF__", "false" if fx_on else "true"), height=0)
