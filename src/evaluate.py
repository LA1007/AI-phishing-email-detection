import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import re

# ========== Configuration ==========
TEXT_COLUMN = 'text_combined'      
LABEL_COLUMN = 'label'             
SAFE_LABEL = 0          
PHISHING_LABEL = 1                 


# Set to True to filter out 'empty' placeholder emails, False to keep them
FILTER_EMPTY_EMAILS = False   # ← change this
# ==================================

# ========== Layer 1: Rule-based Filter ==========
SUSPICIOUS_KEYWORDS = [
    'urgent', 'verify', 'click here', 'suspended', 
    'login now', 'account locked', 'update payment',
    'immediately', 'password expired', 'security alert'
]

def layer1_rule_filter(email_text):
    'Layer 1: Keyword matching'
    if not isinstance(email_text, str):
        return False, []
    
    # SHORT EMAIL EXEMPTION: emails shorter than 20 chars are directly classified as safe
    # (bypasses Layer 2 and Layer 3 entirely)
    if len(email_text) < 20:
        return "safe", []   # Special flag: directly return as Not Phishing
    
    text_lower = email_text.lower()
    matched = [kw for kw in SUSPICIOUS_KEYWORDS if kw in text_lower]
    
    if len(matched) >= 2:
        return True, matched
    return False, matched

def is_mostly_english(text):
    'Check if text is primarily English (English letters > 70% of non-space chars)'
    if not isinstance(text, str):
        return False
    # Count English letters
    letters = re.findall(r'[a-zA-Z]', text)
    # Count total characters (excluding spaces)
    total = len(re.sub(r'\s', '', text))
    if total == 0:
        return False
    return len(letters) / total > 0.7

# ========== Layer 2: Logistic Regression ==========
def load_layer2(model_file, vectorizer_file):
    'Load Layer 2 model and vectorizer'
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_file, "rb") as f:
        _, _, vectorizer = pickle.load(f)
    return model, vectorizer

def layer2_predict(email_text, model, vectorizer):
    'Layer 2: ML prediction, return phishing probability'
    if not isinstance(email_text, str):
        return 0.5
    email_vector = vectorizer.transform([email_text])
    prob = model.predict_proba(email_vector)[0][1]
    return prob


# ========== Layer 3: BERT ==========
def load_layer3(model_path):
    'Load Layer 3 BERT model'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

def layer3_predict(email_text, tokenizer, model):
    'Layer 3: BERT prediction, return phishing probability'
    if not isinstance(email_text, str):
        return 0.5
    inputs = tokenizer(
        email_text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        # Assume label mapping: {0: 'Safe', 1: 'Phishing'}
        phishing_prob = probs[0][1].item()
        # If {0: 'Phishing', 1: 'Safe'}
        # phishing_prob = probs[0][0].item()
    
    return phishing_prob


# ========== Three-layer integrated prediction ==========
def predict_email(email_text, l2_model_file, l2_vectorizer_file, l3_model_path,
                  l2_model=None, l2_vectorizer=None, l3_tokenizer=None, l3_model=None):
    'Three-layer progressive detection'
    
    # Layer 1: Rule-based filter
    is_suspicious, matched_keywords = layer1_rule_filter(email_text)
    if is_suspicious:
        return {
            "prediction": "Phishing",
            "layer": "Layer 1 (Rule)",
            "confidence": None,
            "details": {"matched_keywords": matched_keywords}
        }
    
    # Layer 2
    if l2_model is None:
        l2_model, l2_vectorizer = load_layer2(l2_model_file, l2_vectorizer_file)
    
    l2_prob = layer2_predict(email_text, l2_model, l2_vectorizer)
    
    # Decide whether Layer 3 is needed (threshold 0.6/0.4)
    if l2_prob > 0.6:
        return {
            "prediction": "Phishing",
            "layer": "Layer 2 (High confidence)",
            "confidence": l2_prob,
            "details": {"ml_prob": l2_prob}
        }
    elif l2_prob < 0.4:
        return {
            "prediction": "Not Phishing",
            "layer": "Layer 2 (High confidence)",
            "confidence": l2_prob,
            "details": {"ml_prob": l2_prob}
        }
    else:
        # Layer 3
        if l3_tokenizer is None:
            l3_tokenizer, l3_model = load_layer3(l3_model_path)
        
        l3_prob = layer3_predict(email_text, l3_tokenizer, l3_model)
        result = "Phishing" if l3_prob > 0.5 else "Not Phishing"
        return {
            "prediction": result,
            "layer": "Layer 3 (LLM verdict)",
            "confidence": l3_prob,
            "details": {"ml_prob": l2_prob, "bert_prob": l3_prob}
        }

# ========== Batch evaluation ==========
def evaluate_on_dataset(csv_path, l2_model_file, l2_vectorizer_file, l3_model_path, limit=None):
    'Evaluate three-layer architecture on full dataset and save results to file'
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Data cleaning
    df = df.dropna(subset=[TEXT_COLUMN])
    
    if FILTER_EMPTY_EMAILS:
        df = df[~df[TEXT_COLUMN].str.lower().str.contains('empty')]
    
    df = df[df[TEXT_COLUMN].apply(is_mostly_english)]
    df = df.reset_index(drop=True)
    
    if limit:
        df = df.head(limit)
    
    # Load models
    l2_model, l2_vectorizer = load_layer2(l2_model_file, l2_vectorizer_file)
    l3_tokenizer, l3_model = load_layer3(l3_model_path)
    
    predictions = []
    true_labels = []
    layer_stats = {"Layer 1": 0, "Layer 2": 0, "Layer 3": 0}
    
    # Collect error cases
    false_positives = []
    false_negatives = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        email_text = row[TEXT_COLUMN]
        true_label = row[LABEL_COLUMN]
        
        true_binary = 1 if true_label == PHISHING_LABEL else 0
        
        result = predict_email(
            email_text, 
            l2_model_file, l2_vectorizer_file, l3_model_path,
            l2_model, l2_vectorizer, l3_tokenizer, l3_model
        )
        
        pred_binary = 1 if result['prediction'] == "Phishing" else 0
        predictions.append(pred_binary)
        true_labels.append(true_binary)
        
        if "Layer 1" in result['layer']:
            layer_stats["Layer 1"] += 1
        elif "Layer 2" in result['layer']:
            layer_stats["Layer 2"] += 1
        else:
            layer_stats["Layer 3"] += 1
        
        if true_binary == 0 and pred_binary == 1:
            false_positives.append({
                'index': idx,
                'text': email_text[:500],
                'true_label': true_label,
                'pred_label': result['prediction'],
                'layer': result['layer'],
                'confidence': result['confidence'],
                'details': result['details']
            })
        elif true_binary == 1 and pred_binary == 0:
            false_negatives.append({
                'index': idx,
                'text': email_text[:500],
                'true_label': true_label,
                'pred_label': result['prediction'],
                'layer': result['layer'],
                'confidence': result['confidence'],
                'details': result['details']
            })
    
    # ========== Print to terminal ==========
    print("\n" + "="*60)
    print("Three-layer Architecture Evaluation Results")
    print("="*60)
    
    print("\nLayer processing distribution:")
    total = sum(layer_stats.values())
    for layer, count in layer_stats.items():
        print(f"  {layer}: {count} ({count/total*100:.1f}%)")
    
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=['Safe', 'Phishing']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, predictions)
    print(cm)
    print(f"\nFormat: [[TN, FP], [FN, TP]]")
    
    print(f"\nFalse Positives (legitimate flagged as phishing): {len(false_positives)}")
    print(f"False Negatives (phishing flagged as legitimate): {len(false_negatives)}")
    
    # ========== Save error cases to file ==========
    save_errors_to_file(false_positives, false_negatives, "error_analysis.txt")
    
    return predictions, true_labels, layer_stats, false_positives, false_negatives

def save_errors_to_file(false_positives, false_negatives, output_file="error_analysis.txt"):
    """Save error cases to text file (no console output)"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("Error Case Analysis Report\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"False Positives (legitimate emails flagged as phishing): {len(false_positives)}\n")
        f.write(f"False Negatives (phishing emails flagged as legitimate): {len(false_negatives)}\n\n")
        
        f.write("-"*40 + "\n")
        f.write("False Positive Examples (first 20):\n")
        f.write("-"*40 + "\n")
        for i, fp in enumerate(false_positives[:20]):
            f.write(f"\n--- Example {i+1} (Index: {fp['index']}) ---\n")
            f.write(f"Content: {fp['text']}\n")
            f.write(f"Layer: {fp['layer']}\n")
            f.write(f"Details: {fp['details']}\n")
        
        f.write("\n" + "-"*40 + "\n")
        f.write("False Negative Examples (first 20):\n")
        f.write("-"*40 + "\n")
        for i, fn in enumerate(false_negatives[:20]):
            f.write(f"\n--- Example {i+1} (Index: {fn['index']}) ---\n")
            f.write(f"Content: {fn['text']}\n")
            f.write(f"Layer: {fn['layer']}\n")
            f.write(f"Details: {fn['details']}\n")
    
    # Only print this one line to terminal
    print(f"\nError cases saved to: {output_file}")


# ========== Main ==========
if __name__ == "__main__":
    CSV_PATH = "data/Phishing_Email.csv"
    L2_MODEL_FILE = "models/logistic_regression.pkl"
    L2_VECTORIZER_FILE = "data/preprocessed_data.pkl"
    L3_MODEL_PATH = "models/email_classifier_output/best_model"
    
    predictions, true_labels, layer_stats, false_positives, false_negatives = evaluate_on_dataset(
        CSV_PATH, L2_MODEL_FILE, L2_VECTORIZER_FILE, L3_MODEL_PATH,
        limit=None
    )
