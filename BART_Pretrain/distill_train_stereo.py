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
TEMPERATURE = 2.0  # For distillation
ALPHA = 0.9  # Weight for distillation loss
STUDENT_MODEL = "facebook/bart-base"


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

        if random.choice([True, False]):
            masked_text = f"{head} {tokenizer.mask_token} {tail}"
            full_text = f"{head} {relation} {tail}"
        else:
            masked_text = f"{head} {relation} {tokenizer.mask_token}"
            full_text = f"{head} {relation} {tail}"

        input_string = masked_text + " </s>"
        decoder_input_string = "<s> " + full_text
        labels_string = full_text + " </s>"

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
wandb.init(

)

# Load tokenizer
tokenizer = BartTokenizer.from_pretrained(STUDENT_MODEL)

# Load teacher and student models
teacher_model = BartForConditionalGeneration.from_pretrained("path to COMET BART")
student_model = BartForConditionalGeneration.from_pretrained(STUDENT_MODEL)

# Freeze teacher model parameters
for param in teacher_model.parameters():
    param.requires_grad = False

# Prepare dataset
dataset = TripletBartDataset("path to StereoKG tsv file",tokenizer,)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model.to(device)
student_model.to(device)
lr = 1e-4
optimizer = AdamW(student_model.parameters(), lr=lr)
num_epochs = 1000
num_training_steps = num_epochs * len(dataloader)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Log hyperparameters to wandb
wandb.config.update(
    {
        "learning_rate": lr,
        "num_epochs": num_epochs,
        "batch_size": BATCH_SIZE,
        "temperature": TEMPERATURE,
        "alpha": ALPHA,
    }
)

# Loss functions
distillation_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")

# Training loop
student_model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Teacher model output
        with torch.no_grad():
            teacher_outputs = teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )
            teacher_logits = teacher_outputs.logits / TEMPERATURE

        # Student model output
        student_outputs = student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        student_logits = student_outputs.logits / TEMPERATURE
        teacher_logits = teacher_outputs.logits[..., : student_logits.size(-1)]

        # Compute losses
        distillation_loss = distillation_loss_fn(
            torch.nn.functional.log_softmax(student_logits, dim=-1),
            torch.nn.functional.softmax(teacher_logits, dim=-1),
        )
        supervised_loss = student_outputs.loss

        # Combine losses
        loss = ALPHA * distillation_loss + (1 - ALPHA) * supervised_loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Update epoch loss
        epoch_loss += loss.item()

        # Log losses to wandb
        wandb.log(
            {
                "loss": loss.item(),
                "distillation_loss": distillation_loss.item(),
                "supervised_loss": supervised_loss.item(),
            }
        )

    # Log average epoch loss
    wandb.log({"epoch_loss": epoch_loss / len(dataloader)})

    if epoch % 100 == 0:
        alpha_save = int(ALPHA * 10)
        checkpoint_dir = f"directory to save checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        student_model.save_pretrained(checkpoint_dir)
        torch.save(student_model.state_dict(), f"{checkpoint_dir}/pytorch_model.bin")
        tokenizer.save_pretrained(checkpoint_dir)

# Finish wandb run
wandb.finish()
