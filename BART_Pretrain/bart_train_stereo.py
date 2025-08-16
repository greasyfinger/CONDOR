import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import random
from tqdm import tqdm
import wandb
import csv

BATCH_SIZE = 32


class TripletBartDataset(Dataset):
    def __init__(self, tsv_file, tokenizer, max_length=24):
        self.triplets = self.load_tsv(tsv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_tsv(self, tsv_file):
        with open(tsv_file, "r", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            return list(reader)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        head, relation, tail = self.triplets[idx]

        # Randomly choose to mask either relation or tails
        if random.choice([True, False]):
            masked_text = f"{head} {tokenizer.mask_token} {tail}"
            full_text = f"{head} {relation} {tail}"
        else:
            masked_text = f"{head} {relation} {tokenizer.mask_token}"
            full_text = f"{head} {relation} {tail}"

        # masked_text = f"{head} {relation} <mask>"
        # full_text = f"{head} {relation} {tail}"

        input_string = masked_text + " </s>"
        decoder_input_string = "<s> " + full_text
        labels_string = full_text + " </s>"

        # Tokenize and pad
        input_encodings = self.tokenizer(
            input_string,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        decoder_encodings = self.tokenizer(
            decoder_input_string,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        label_encodings = self.tokenizer(
            labels_string,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": input_encodings.input_ids.squeeze(),
            "attention_mask": input_encodings.attention_mask.squeeze(),
            "decoder_input_ids": decoder_encodings.input_ids.squeeze(),
            "decoder_attention_mask": decoder_encodings.attention_mask.squeeze(),
            "labels": label_encodings.input_ids.squeeze(),
        }


class CustomBartDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        words = text.split()

        # Create input with masks
        mask_indices = random.sample(
            range(len(words)), k=max(1, int(len(words) * 0.15))
        )
        input_words = words.copy()
        for i in mask_indices:
            input_words[i] = "<mask>"
        input_string = " ".join(input_words) + " </s>"

        # Create decoder input and labels
        decoder_input_string = "<s> " + text
        labels_string = text + " </s>"

        # Tokenize and pad
        input_encodings = self.tokenizer(
            input_string,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        decoder_encodings = self.tokenizer(
            decoder_input_string,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        label_encodings = self.tokenizer(
            labels_string,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": input_encodings.input_ids.squeeze(),
            "attention_mask": input_encodings.attention_mask.squeeze(),
            "decoder_input_ids": decoder_encodings.input_ids.squeeze(),
            "decoder_attention_mask": decoder_encodings.attention_mask.squeeze(),
            "labels": label_encodings.input_ids.squeeze(),
        }


# Initialize wandb
wandb.init()

# Load tokenizer and initialize model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

dataset = TripletBartDataset(
    "path to StereKG tsv file",
    tokenizer,
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-4)
num_epochs = 12
num_training_steps = num_epochs * len(dataloader)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Log hyperparameters to wandb
wandb.config.update(
    {"learning_rate": 5e-5, "num_epochs": num_epochs, "batch_size": BATCH_SIZE}
)

# Training loop
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Update epoch loss
        epoch_loss += loss.item()

        # Log loss to wandb
        wandb.log({"loss": loss.item()})

    # Log average epoch loss to wandb
    wandb.log({"epoch_loss": epoch_loss / len(dataloader)})

    if epoch >= 10:
        checkpoint_dir = f""
        os.makedirs(checkpoint_dir, exist_ok=True)

        model.save_pretrained(checkpoint_dir)
        torch.save(model.state_dict(), f"{checkpoint_dir}/pytorch_model.bin")
        tokenizer.save_pretrained(checkpoint_dir)

# Finish wandb run
wandb.finish()
