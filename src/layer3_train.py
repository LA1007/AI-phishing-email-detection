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
    EarlyStoppingCallback,
)
from transformers.utils.notebook import NotebookProgressCallback
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
    train_df, val_df, test_df = stratified_split(df, SEED)

    print_split_stats("Train", train_df)
    print_split_stats("Validation", val_df)
    print_split_stats("Test", test_df)

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "Safe", 1: "Phishing"},
        label2id={"Safe": 0, "Phishing": 1},
    )

    train_dataset = EmailDataset(
        texts=train_df["Email Text"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=512,
    )

    val_dataset = EmailDataset(
        texts=val_df["Email Text"].tolist(),
        labels=val_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=512,
    )

    test_dataset = EmailDataset(
        texts=test_df["Email Text"].tolist(),
        labels=test_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=512,
    )

    training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",
    disable_tqdm=True,
    fp16=torch.cuda.is_available()
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    
    trainer.remove_callback(NotebookProgressCallback)
    trainer.train()
    
    val_metrics = trainer.evaluate(eval_dataset=val_dataset)
    print("Validation metrics:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v}")

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

    best_model_dir = os.path.join(OUTPUT_DIR, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    pred_df = test_df.copy()
    pred_df["true_label"] = pred_df["label"].map({0: "Safe", 1: "Phishing"})
    pred_df["predicted_label"] = [("Phishing" if p == 1 else "Safe") for p in test_preds]
    pred_df["correct"] = pred_df["true_label"] == pred_df["predicted_label"]
    pred_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)