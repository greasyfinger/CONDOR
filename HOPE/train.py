import os
from datasets import load_from_disk
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="facebook/bart-base",
    help="model path",
)
args = parser.parse_args()

MODEL = args.model
EXP = os.path.basename(MODEL)

# Load pre-processed datasets
train_dataset = load_from_disk("./conv_data/therapy_train")
val_dataset = load_from_disk("./conv_data/therapy_val")

# Initialize tokenizer and model
tokenizer = BartTokenizer.from_pretrained(MODEL)
model = BartForConditionalGeneration.from_pretrained(MODEL)


# Preprocessing function
def preprocess_function(examples):
    inputs = [text for text in examples["input_text"]]
    targets = [text for text in examples["target_text"]]

    model_inputs = tokenizer(
        inputs, max_length=512, truncation=True, padding="max_length"
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=128, truncation=True, padding="max_length"
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Apply preprocessing
tokenized_train = train_dataset.map(
    preprocess_function, batched=True, remove_columns=["input_text", "target_text"]
)

tokenized_val = val_dataset.map(
    preprocess_function, batched=True, remove_columns=["input_text", "target_text"]
)

wandb.init(project="Therapy_bart", name=EXP)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=f"",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    num_train_epochs=10,
    fp16=True,
    logging_dir=f"",
    logging_steps=100,
    report_to="wandb",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    run_name=EXP,
)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Start training
print("Starting training...")
trainer.train()

# Save final model
trainer.save_model(f"")
print(f"Training complete. Model saved to heal_bart/{EXP}")
