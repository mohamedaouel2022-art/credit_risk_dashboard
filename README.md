# CrediScope — Credit Risk Intelligence

A polished Streamlit prototype for credit-risk assessment, portfolio exploration, and model validation. It is designed to work with the `credit_risk_dataset.csv` used in the project.

> **Important:** This is an educational/research prototype, not a regulated lending decision system. The predicted score is not guaranteed to be a calibrated probability of default. Do not use it as the sole basis for real lending decisions.

## 1. Project structure

```text
crediscope_credit_risk_platform/
├── app.py                    # Streamlit user interface
├── train_model.py             # Preprocessing, model comparison, evaluation, artifacts
├── requirements.txt
├── README.md
├── .gitignore
├── credit_risk_dataset.csv    # Add your own dataset here
├── credit_risk_model.joblib   # Created by training
└── model_metrics.json         # Created by training
```

## 2. Install on macOS

Open Terminal and move into this folder. If you downloaded/extracted the ZIP to Downloads, for example:

```bash
cd ~/Downloads/crediscope_credit_risk_platform
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Add your dataset

Copy `credit_risk_dataset.csv` into this folder. The target column must be named `loan_status` and use:

- `0` = non-default
- `1` = default

If your file is elsewhere, pass its full path when training. Example based on the previous project path:

```bash
python train_model.py --data "/Users/macuser/Desktop/credit_risk/credit_risk_dataset.csv"
```

Or copy the file here and run:

```bash
python train_model.py
```

Training compares Logistic Regression and Random Forest on a stratified holdout split, then saves the best by ROC-AUC. The default reporting threshold is 0.35; change it explicitly with `--threshold 0.40` if you want to inspect another operating point. Threshold choice should ultimately be based on validated business costs and policy, not convenience.

## 4. Launch the web app

```bash
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). Open it in your browser.

## 5. Included pages

- **Executive Overview:** dataset-level KPIs and distributions.
- **Credit Assessment:** interactive application form and model score.
- **Portfolio Analytics:** exposure and observed outcomes by segment.
- **Model Intelligence:** holdout metrics, confusion matrix, and governance checklist.
- **Data Explorer:** preview, schema, and missingness.

## 6. Modeling notes

- Preprocessing is fitted inside each scikit-learn pipeline to reduce leakage from preprocessing before the split.
- Numeric fields use median imputation and scaling; categorical fields use most-frequent imputation and one-hot encoding.
- Models are compared on holdout ROC-AUC; the holdout set should not be repeatedly used for model tuning.
- The interface shows the model's positive-class score. It should not be called a calibrated probability of default unless calibration has been evaluated and, if needed, applied on separate data.
- Review feature meaning carefully. For example, interest rate and loan grade may encode prior underwriting decisions. They may be inappropriate for some prospective scoring use cases.
- Before production, add temporal/out-of-time validation, calibration testing, fairness and compliance reviews, access control, audit logging, monitoring, and human review.

## Troubleshooting

**`Dataset not found`** — put the CSV next to `train_model.py`, or pass the full path with `--data`.

**`ModuleNotFoundError`** — activate `.venv` and run `pip install -r requirements.txt` again.

**`streamlit: command not found`** — run `python -m streamlit run app.py` after activating the environment.

**Assessment schema error** — retrain the model after replacing or changing the dataset. The model artifact and `model_metrics.json` must come from the same training run.
