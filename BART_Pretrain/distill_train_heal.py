import torch.nn.functional as F
import torch
from transformers import (
    DataCollatorForSeq2Seq,
    BartTokenizer,
    BartForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import pandas as pd
import argparse

# Parse command-line argument for ALPHA
parser = argparse.ArgumentParser()
parser.add_argument(
    "--alpha",
    type=float,
    default=0.5,
    help="Weighting factor for loss combination (default: 0.5)",
)
parser.add_argument(
    "--coverage",
    type=float,
    default=1,
    help="percentage of graph you want to use",
)
args = parser.parse_args()

# Assign to global ALPHA (minimal change to original code)
ALPHA = args.alpha

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
comet_bart = BartForConditionalGeneration.from_pretrained(
    "path to comet"
).to(device)
comet_bart.eval()

model = BartForConditionalGeneration.from_pretrained(
    "path to base model"
).to(device)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits

        # Standard cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        ce_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        # KL Divergence loss
        with torch.no_grad():
            comet_logits = comet_bart(**inputs).logits[
                ..., : logits.size(-1)
            ]  # Reference model's logits

        kl_loss = F.kl_div(
            F.log_softmax(logits, dim=-1),
            F.softmax(comet_logits, dim=-1),
            reduction="batchmean",
        )

        total_loss = ALPHA * ce_loss + (1 - ALPHA) * kl_loss

        return (total_loss, outputs) if return_outputs else total_loss


data = pd.read_csv("path to stressors and responses")
if args.coverage < 1.0:
    num_rows = int(len(data) * args.coverage)
    data = data.iloc[:num_rows]

# split the data into train and test and convert the dataframe into a Dataset object

train_data = data.iloc[: int(0.8 * len(data))]
val_data = data.iloc[int(0.8 * len(data)) :]

train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)


def preprocess_function(examples):
    """preprocess data"""

    model_inputs = tokenizer(
        examples["Story"],
        max_length=1024,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    labels = tokenizer(
        text_target=examples["Response"],
        max_length=256,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["decoder_attention_mask"] = labels["attention_mask"]
    return model_inputs


tokenized_train_dataset = train_dataset.map(
    preprocess_function, batched=True, remove_columns=train_dataset.column_names
)
tokenized_val_dataset = val_dataset.map(
    preprocess_function, batched=True, remove_columns=val_dataset.column_names
)

# define training arguments


data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8,  # Optional but recommended for GPU efficiency
)

training_args = TrainingArguments(
    output_dir=f"output dir",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    num_train_epochs=15,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=500,
    logging_dir=f"log dir",
    logging_steps=10,
    fp16=False,
)

# define trainer object

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator,
)

# fine tune the model

trainer.train()

# save the model

model.save_pretrained(f"save model")
tokenizer.save_pretrained(
    f"save tokenizer"
)
