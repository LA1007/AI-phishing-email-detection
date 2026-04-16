# AI-Powered Phishing Email Detection

A hybrid phishing email detection system using a **three-layer progressive architecture**:
- **Layer 1**: Rule-based keyword filtering
- **Layer 2**: Logistic Regression with TF-IDF + Bigram features
- **Layer 3**: BERT semantic verification (DistilBERT) for ambiguous cases

Overall accuracy: **99%** on the filtered dataset.

---

## 📌 Project Overview

This project detects phishing emails using a layered approach that balances speed, cost, and accuracy:

| Layer | Method | Role |
|-------|--------|------|
| **Layer 1** | Rule-based keyword matching + whitelist | Quick filtering (0.9% of emails) |
| **Layer 2** | TF-IDF (unigram + bigram) + Logistic Regression | Main workhorse (96.9% of emails) |
| **Layer 3** | DistilBERT semantic understanding | Final judge for edge cases (2.2% of emails) |

---

## 📊 Results

### Model Comparison (Layer 2)

| Model | Accuracy | Training Time |
|-------|----------|---------------|
| **Logistic Regression** | **98.19%** | **2.43s** |
| Random Forest | 97.22% | 17.18s |
| Calibrated RF | 97.19% | 65.01s |
| XGBoost | 95.61% | 20.67s |

### Three-Layer Overall Performance

| Metric | Value |
|--------|-------|
| **Overall accuracy** | **99%** |
| False Positives | 90 |
| False Negatives | 59 |
| Layer 1 handled | 1.0% |
| Layer 2 handled | 96.9% |
| Layer 3 handled | 2.2% |

---

## 🛠️ Installation

```bash
git clone https://github.com/LA1007/AI-phishing-email-detection.git
cd AI-phishing-email-detection
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

If requirements.txt is not available, install manually:

bash
pip install pandas scikit-learn transformers torch tqdm xgboost
```

---

🧪 Usage
Run these commands in order:

```bash
python src/preprocess.py
python src/train_models.py
python src/evaluate.py
```
To test a single email:

```bash
python src/predict.py
```

---

⚙️ Configuration
At the top of preprocess.py and evaluate.py:

| Setting | Options | Description |
|-------|----------|---------------|
| **FILTER_EMPTY_EMAILS** | **True / False** | **Remove "empty" placeholder emails (recommended: True)** |

Note: Use FILTER_EMPTY_EMAILS = True unless you have a specific reason to keep placeholders, since "empty" entries are data artifacts, not real-world emails.

---

📁 Dataset Format
Place your dataset as data/Phishing_Email.csv with:

| Column | Values |
|-------|----------|
| **Email Text** | **Email content** |
| **Email Type** | **Safe Email or Phishing Email** |

To customize column names, edit the config at the top of each Python file.
