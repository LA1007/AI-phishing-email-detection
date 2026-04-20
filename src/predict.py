import pickle
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# ========== Configuration ==========
L2_MODEL_FILE = "models/logistic_regression.pkl"
L2_VECTORIZER_FILE = "data/preprocessed_data.pkl"
L3_MODEL_PATH = "models/email_classifier_output/best_model"
# ==================================

# ========== Layer 1: Rule-based Filter ==========
SUSPICIOUS_KEYWORDS = [
    'urgent', 'verify', 'click here', 'suspended', 
    'login now', 'account locked', 'update payment',
    'immediately', 'password expired', 'security alert'
]
WHITELIST_KEYWORDS = ['newsletter', 'subscriber', 'unsubscribe']

def layer1_rule_filter(email_text):
    'Layer 1: Keyword matching'
    if not isinstance(email_text, str):
        return False, []
    
    # Short email exemption (less than 20 characters, likely normal chat)
    if len(email_text) < 20:
        return False, []
    
    text_lower = email_text.lower()
    matched = [kw for kw in SUSPICIOUS_KEYWORDS if kw in text_lower]
    
    if len(matched) >= 2:
        return True, matched
    return False, matched

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
    print(f"Layer 3 label mapping: {model.config.id2label}")
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
    
    return phishing_prob

# ========== Three-layer integrated prediction ==========
def predict_email(email_text, l2_model_file, l2_vectorizer_file, l3_model_path,
                  l2_model=None, l2_vectorizer=None, l3_tokenizer=None, l3_model=None):
    'Three-layer progressive detection'
    
    # Layer 1: Rule-based filter
    result = layer1_rule_filter(email_text)
    
    # Handle short email exemption
    if result[0] == "safe":
        return {
            "prediction": "Not Phishing",
            "layer": "Layer 1 (Short email exemption)",
            "confidence": None,
            "details": {"reason": "email shorter than 20 characters"}
        }
    
    is_suspicious, matched_keywords = result
    if is_suspicious:
        return {
            "prediction": "Phishing",
            "layer": "Layer 1 (Rule)",
            "confidence": None,
            "details": {"matched_keywords": matched_keywords}
        }
    
    # Load Layer 2 if not passed in
    if l2_model is None:
        l2_model, l2_vectorizer = load_layer2(l2_model_file, l2_vectorizer_file)
    
    # Layer 2: ML prediction
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
        # Load Layer 3 if not passed in
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

# ========== Main ==========
if __name__ == "__main__":
    # Test emails
    test_emails = [
        "Your account has been compromised. Click here to reset your password.",
        "Meeting tomorrow at 3 PM. Please confirm your attendance.",
        "Urgent! Your PayPal account has been suspended. Verify now to avoid closure.",
        "Hi, just checking in. How was your weekend?",
        "Dear customer, your bank account has been locked due to suspicious activity. Click https://fake-bank.com to unlock."
    ]
    
    print("=" * 60)
    print("Three-layer Phishing Email Detection System Test")
    print("=" * 60)
    
    for i, email in enumerate(test_emails, 1):
        print(f"\nTest {i}:")
        print(f"Email: {email}")
        print("-" * 40)
        
        result = predict_email(email, L2_MODEL_FILE, L2_VECTORIZER_FILE, L3_MODEL_PATH)
        
        print(f"Prediction: {result['prediction']}")
        print(f"Layer: {result['layer']}")
        if result['confidence']:
            print(f"Confidence: {result['confidence']:.4f}")
        print(f"Details: {result['details']}")
        print("-" * 40)
    
    print("\n" + "=" * 60)
    print("Test completed")
    print("=" * 60)
