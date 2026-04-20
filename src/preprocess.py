import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re

# ========== Configuration ==========
TEXT_COLUMN = 'Email Text'
LABEL_COLUMN = 'Email Type'
SAFE_LABEL = 'Safe Email'
PHISHING_LABEL = 'Phishing Email'

# ========== DATA CLEANING OPTION ==========
# Set to True to filter out 'empty' placeholder emails, False to keep them
FILTER_EMPTY_EMAILS = False   # ← change this
# =========================================

def is_mostly_english(text):
    'Check if text is primarily English (English letters > 70% of non-space chars)'
    if not isinstance(text, str):
        return False
    letters = re.findall(r'[a-zA-Z]', text)
    total = len(re.sub(r'\s', '', text))
    if total == 0:
        return False
    return len(letters) / total > 0.7

def preprocess_data(input_file, output_file):
    data = pd.read_csv(input_file)
    
    # 1. Drop null values
    data = data.dropna(subset=[TEXT_COLUMN])
    
    # 2. Optionally filter out 'empty' placeholder emails
    if FILTER_EMPTY_EMAILS:
        data = data[~data[TEXT_COLUMN].str.lower().str.contains('empty')]
        print("Filtered out emails containing 'empty'")
    else:
        print("Kept emails containing 'empty'")
    
    # 3. Filter out non-English emails
    data = data[data[TEXT_COLUMN].apply(is_mostly_english)]
    
    data = data.reset_index(drop=True)

    # Vectorize
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    X = vectorizer.fit_transform(data[TEXT_COLUMN]).toarray()
    y = data[LABEL_COLUMN].map({SAFE_LABEL: 0, PHISHING_LABEL: 1}).values

    with open(output_file, "wb") as f:
        pickle.dump((X, y, vectorizer), f)

    print(f"Preprocessed data saved to {output_file}")
    print(f"Processed {len(data)} emails with {X.shape[1]} features")

if __name__ == "__main__":
    preprocess_data("data/Phishing_Email.csv", "data/preprocessed_data.pkl")
