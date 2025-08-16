import os
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    BartForConditionalGeneration,
    BartForSequenceClassification,
)
from prepare import get_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict, List
import argparse
import os

# Argument parser setup
parser = argparse.ArgumentParser(
    description="Train BART classifier for sexism detection"
)

parser.add_argument(
    "--model",
    type=str,
    default="",
    help="Path to the pretrained model",
)
parser.add_argument("--run", type=str, default="", help="WandB run name")
parser.add_argument(
    "--wandb_project", type=str, default="", help="WandB project name"
)
parser.add_argument("--cuda_device", type=str, default="0", help="CUDA visible devices")
parser.add_argument(
    "--num_epochs", type=int, default=6, help="Number of training epochs"
)
parser.add_argument(
    "--batch_size", type=int, default=32, help="Batch size for training and evaluation"
)

args = parser.parse_args()

# Apply environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
os.environ["WANDB_PROJECT"] = args.wandb_project

MODEL = args.model
RUN = args.run


gen_model = BartForConditionalGeneration.from_pretrained(MODEL)

config = gen_model.config
config.num_labels = 4

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, use_threads=False)
cls_model = BartForSequenceClassification(config)
cls_model.model.encoder.load_state_dict(gen_model.model.encoder.state_dict())
cls_model.resize_token_embeddings(len(tokenizer))


def prepare_data(
    train_df: pd.DataFrame, dev_df: pd.DataFrame, test_df: pd.DataFrame, task="B"
) -> Dict[str, Dataset]:
    """
    Convert pandas dataframes to HuggingFace datasets
    """
    if task == "A":
        label_names = ["not sexist", "sexist"]
    elif task == "B":
        label_names = [
            "animosity",
            "derogation",
            "prejudiced discussions",
            "threats, plans to harm and incitement",
        ]
    label2id = {label: idx for idx, label in enumerate(label_names)}
    id2label = {idx: label for label, idx in label2id.items()}

    # Convert labels to integers
    for df in [train_df, dev_df, test_df]:
        df["label"] = df["label"].map(label2id)

    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)
    test_dataset = Dataset.from_pandas(test_df)

    return (
        {"train": train_dataset, "validation": dev_dataset, "test": test_dataset},
        label2id,
        id2label,
    )


def compute_metrics(pred_obj):
    """
    Compute metrics for model evaluation
    """
    predictions = (
        pred_obj.predictions[0]
        if isinstance(pred_obj.predictions, tuple)
        else pred_obj.predictions
    )
    preds = np.argmax(predictions, axis=1)
    labels = pred_obj.label_ids

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    acc = accuracy_score(labels, preds)

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def train_bart_classifier(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = f"",
    num_epochs: int = 3,
    batch_size: int = 16,
    task="B",
):
    """
    Train BART model for sexism classification
    """
    # Prepare datasets
    datasets, label2id, id2label = prepare_data(train_df, dev_df, test_df, task=task)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=512, padding=False
        )

    # Tokenize datasets with no batching to avoid parallelism issues
    tokenized_datasets = {
        split: dataset.map(
            tokenize_function, batched=False, desc=f"Tokenizing {split} dataset"
        )
        for split, dataset in datasets.items()
    }

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        save_total_limit=1,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False,
        dataloader_num_workers=0,
        report_to="wandb",
        run_name=RUN,
    )

    # Initialize trainer
    trainer = Trainer(
        model=cls_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    # Train model
    trainer.train()


# Example usage
if __name__ == "__main__":
    # Assuming you have your dataframes ready
    task = "B"
    train_df, dev_df, test_df = get_dataset(task=task)

    trainer, test_results = train_bart_classifier(
        train_df=train_df,
        dev_df=dev_df,
        test_df=test_df,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        task=task,
    )

    print("Test Results:", test_results)
