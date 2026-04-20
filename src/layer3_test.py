import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from layer3_helpers import (
    validate_and_prepare_dataframe,
    EmailDataset,
    compute_metrics,
    stratified_split,
    print_split_stats,
    clean_csv,
    DATA_DIR,
    MODELS_DIR
)

if __name__ == "__main__":
    MODEL_DIR = MODELS_DIR / "email_classifier_output/best_model"
    OUTPUT_DIR = MODELS_DIR / "email_classifier_output"
    MODEL_NAME = "distilbert-base-uncased" #DistilBERT selected for better efficiency and better performance according to https://www.mdpi.com/1999-4893/18/10/599
    SEED = 42
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #https://www.kaggle.com/datasets/subhajournal/phishingemails/data
    df1 = pd.read_csv(DATA_DIR / "Regular Phishing Email Dataset/Phishing_Email.csv")
    df1 = validate_and_prepare_dataframe(df1)
    
    #https://www.kaggle.com/datasets/francescogreco97/human-llm-generated-phishing-legitimate-emails
    df2 = clean_csv(DATA_DIR / "AI Phishing Email Dataset/llm-generated/phishing.csv", "phishing email")
    df2 = validate_and_prepare_dataframe(df2)
    
    df3 = clean_csv(DATA_DIR / "AI Phishing Email Dataset/llm-generated/legit.csv", "safe email")
    df3 = validate_and_prepare_dataframe(df3)

    df = pd.concat([df1, df2, df3], ignore_index=True)
    print(f"Total cleaned samples: {len(df)}")

    train_df, val_df, test_df = stratified_split(df, random_state=SEED)

    print_split_stats("Train", train_df)
    print_split_stats("Validation", val_df)
    print_split_stats("Test", test_df)

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)

    test_dataset = EmailDataset(
        texts=test_df["Email Text"].tolist(),
        labels=test_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=512,
    )

    eval_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_eval_batch_size=16,
        report_to="none",
        disable_tqdm=True,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        compute_metrics=compute_metrics,
    )

    test_output = trainer.predict(test_dataset)
    test_preds = np.argmax(test_output.predictions, axis=1)
    test_labels = test_output.label_ids

    test_accuracy = accuracy_score(test_labels, test_preds)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        test_labels,
        test_preds,
        average="binary",
        zero_division=0,
    )

    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1-score: {test_f1:.4f}")

    print(
        classification_report(
            test_labels,
            test_preds,
            target_names=["Safe", "Phishing"],
            zero_division=0,
        )
    )

    pred_df = test_df.copy()
    pred_df["true_label"] = pred_df["label"].map({0: "Safe", 1: "Phishing"})
    pred_df["predicted_label"] = ["Phishing" if p == 1 else "Safe" for p in test_preds]
    pred_df["correct"] = pred_df["true_label"] == pred_df["predicted_label"]

    predictions_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    pred_df.to_csv(predictions_path, index=False)