import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

def train_all_models(data_file):
    # Load data
    with open(data_file, "rb") as f:
        X, y, vectorizer = pickle.load(f)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("="*60)
    print("Model Comparison Experiment")
    print("="*60)
    print(f"Training set: {X_train.shape[0]} samples, features: {X_train.shape[1]}")
    print(f"Test set: {X_test.shape[0]} samples")
    print("="*60)
    
    results = {}
    
    # ========== 1. Random Forest  ==========
    print("\n[1/4] Training Random Forest (Baseline)...")
    start = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_time = time.time() - start
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    results['Random Forest'] = {
        'accuracy': rf_acc,
        'time': rf_time,
        'predictions': rf_pred,
        'model': rf
    }
    print(f"    Accuracy: {rf_acc:.4f}, Time: {rf_time:.2f}s")
    
    # ========== 2. Calibrated Random Forest ==========
    print("\n[2/4] Training Calibrated Random Forest (isotonic)...")
    start = time.time()
    rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
    calibrated_rf = CalibratedClassifierCV(rf_base, cv=5, method='isotonic')
    calibrated_rf.fit(X_train, y_train)
    cal_rf_time = time.time() - start
    cal_rf_pred = calibrated_rf.predict(X_test)
    cal_rf_acc = accuracy_score(y_test, cal_rf_pred)
    results['Calibrated RF'] = {
        'accuracy': cal_rf_acc,
        'time': cal_rf_time,
        'predictions': cal_rf_pred,
        'model': calibrated_rf
    }
    print(f"    Accuracy: {cal_rf_acc:.4f}, Time: {cal_rf_time:.2f}s")
    
    # ========== 3. Logistic Regression ==========
    print("\n[3/4] Training Logistic Regression...")
    start = time.time()
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_time = time.time() - start
    lr_pred = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    results['Logistic Regression'] = {
        'accuracy': lr_acc,
        'time': lr_time,
        'predictions': lr_pred,
        'model': lr
    }
    print(f"    Accuracy: {lr_acc:.4f}, Time: {lr_time:.2f}s")
    
    # ========== 4. XGBoost  ==========
    try:
        import xgboost as xgb
        print("\n[4/4] Training XGBoost...")
        start = time.time()
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        xgb_time = time.time() - start
        xgb_pred = xgb_model.predict(X_test)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        results['XGBoost'] = {
            'accuracy': xgb_acc,
            'time': xgb_time,
            'predictions': xgb_pred,
            'model': xgb_model
        }
        print(f"    Accuracy: {xgb_acc:.4f}, Time: {xgb_time:.2f}s")
    except ImportError:
        print("\n[4/4] XGBoost not installed, skipping")
    
    # ========== Summary comparison ==========
    print("\n" + "="*60)
    print("Model Comparison Summary")
    print("="*60)
    print(f"{'Model':<25} {'Accuracy':<10} {'Time(s)':<10}")
    print("-"*45)
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['accuracy']:.4f}     {metrics['time']:.2f}")
    
    # Save all models
    for name, metrics in results.items():
        model_file = f"models/{name.lower().replace(' ', '_')}.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(metrics['model'], f)
        print(f"\n{name} model saved to: {model_file}")
    
    # Save vectorizer (for prediction use)
    with open("models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print("\nVectorizer saved to: models/vectorizer.pkl")
    
    return results, X_test, y_test

if __name__ == "__main__":
    results, X_test, y_test = train_all_models("data/preprocessed_data.pkl")
