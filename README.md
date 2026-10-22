# ThreatScan AI

**A local-first fraud, phishing, and scam risk scanner** built for the IBM SkillsBuild Cyber Security 2026 project submission.

ThreatScan AI analyzes message text and URLs with six machine-learning models, URL and language features, and a rule-based signal layer. It presents a risk score, verdict, model signals, URL findings, and safety recommendations in a Streamlit dashboard.

## Submission files

- **Code:** [`Threat_scan.ai.py`](./Threat_scan.ai.py) — consolidated, standalone Python application, including the sample dataset, feature engineering, training, inference, and web UI.
- **Dependencies:** [`requirements.txt`](./requirements.txt)
- **Project report:** [`ThreatScan_ProjectReport.docx`](./ThreatScan_ProjectReport.docx)
- **Dataset:** [`dataset.csv`](./dataset.csv) — the project’s embedded 156-example sample corpus. This is a locally prepared educational dataset; it is not sourced from an external dataset provider.

## Technology

- Python and Streamlit
- scikit-learn, NumPy, and joblib
- TF-IDF with Logistic Regression
- Character n-gram TF-IDF with calibrated LinearSVC
- Random Forest and Gradient Boosting
- Multilayer Perceptron (MLP)
- Multinomial Naive Bayes
- Handcrafted text, URL, and linguistic indicators

## Setup and run

Use Python 3.11 or newer. From this project directory, install the pinned dependencies:

```powershell
python -m pip install -r requirements.txt
```

Train the models once (and again after changing the scikit-learn version):

```powershell
python Threat_scan.ai.py --train
```

Start the dashboard:

```powershell
python -m streamlit run Threat_scan.ai.py
```

The app opens at <http://localhost:8501>. Model files are written to the `models/` folder next to the script. The checked-in model artifacts were trained with the scikit-learn version pinned in `requirements.txt`.

Use the same Python version for installing dependencies, training, and launching the app. Avoid launching with a standalone `streamlit` command if it may resolve to a different Python installation. If you select Python explicitly with the Windows launcher, use the same selector for each command (for example, `py -3.12 -m pip install -r requirements.txt` and `py -3.12 -m streamlit run Threat_scan.ai.py`).

## How it works

1. Enter a message or choose an included demonstration sample.
2. The scanner extracts 36 handcrafted text and URL features and evaluates the message with six trained classifiers.
3. A weighted ensemble combines the model outputs with URL-risk and rule-based signals.
4. The dashboard displays the verdict, risk score, model breakdown, URL indicators, and recommendations.

Ensemble weights: TF-IDF Logistic Regression 25%, character n-gram SVC 20%, Random Forest 18%, Gradient Boosting 17%, MLP 12%, and Naive Bayes 8%.

## Dataset and limitations

The 156 labeled examples (92 fraud/scam and 64 legitimate) are embedded directly in `Threat_scan.ai.py`, so training does not depend on a network connection or an external dataset URL. When you run `python Threat_scan.ai.py --train`, the script creates `dataset.csv` beside itself if that CSV is missing; it preserves an existing CSV rather than overwriting it. The CSV is an inspectable export of the embedded training corpus.

The held-out metrics in the report are from this small corpus only and should not be interpreted as real-world detection accuracy. The scanner is a learning and awareness aid—not a substitute for security tools, provider verification, or expert review. Treat uncertain results cautiously and do not use the app as the sole basis for a security decision.

## Privacy

Message analysis runs locally; the app does not send message contents to an analysis API. No network lookup is performed for submitted URLs. The application is intended for local use.
