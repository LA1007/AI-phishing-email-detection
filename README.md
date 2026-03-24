# AI-Powered Phishing Email Detection

A hybrid phishing email detection system using **TF-IDF + Random Forest** (Layer 2) with planned integration of **rule-based filtering** (Layer 1) and **LLM-based semantic verification** (Layer 3).

## 📌 Project Overview

This project aims to detect phishing emails using a layered approach:
- **Layer 1 (Rule-based)**: Quick filtering using keyword matching
- **Layer 2 (ML)**: TF-IDF vectorization + Random Forest classifier
- **Layer 3 (LLM)**: Semantic analysis using transformers (BERT/DistilBERT) for ambiguous cases

## 📊 Dataset

Two datasets were used:
- Dataset 1: 18,634 emails from Kaggle
- Dataset 2: 4,000 emails (human + LLM-generated) from ITASEC 2024 paper

## 🚀 Results

| Dataset | Features | Accuracy |
|---------|----------|----------|
| Dataset 1 (limited) | 5,000 | 96.32% |
| Dataset 1 (full) | 163,224 | 96.54% |
| Dataset 2 | 38,803 | **99.12%** |

## 🛠️ Installation

```bash
git clone https://github.com/LA1007/AI-phishing-email-detection.git
cd AI-phishing-email-detection
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt