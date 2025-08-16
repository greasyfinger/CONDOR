# make imports

import os
import torch
from transformers import DataCollatorForSeq2Seq, BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# load data

def load_data(data_dir):

    """Load data"""

    data = []

    for file in os.listdir(data_dir):

        temp_fp = os.path.join(data_dir, file)
        df = pd.read_csv(temp_fp)

        summary = df.iloc[-3]['Sub topic']
        utterances = df.iloc[:-3]['Utterance'].tolist()

        # convert all utterances into string format
        for i in range(len(utterances)):
            utterances[i] = str(utterances[i])

        data.append((utterances, summary))

    # convert list of tuples into a dictionary
    formatted_data = {
    'utterances': [' '.join(utterances) for utterances, _ in data],
    'summary': [summary for _, summary in data]
    }
 
    return formatted_data

# train_data = load_data('Data/Train')
# val_data = load_data('Data/Validation')

# convert the dictionary into a Dataset object

train_dataset = Dataset.from_dict(train_data)
val_dataset = Dataset.from_dict(val_data)

# load tokenizer and preprocess data

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

def preprocess_function(examples):

    """preprocess data"""
    
    model_inputs = tokenizer(
        examples['utterances'],
        max_length=1024,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    labels = tokenizer(
        text_target=examples['summary'],
        max_length=256,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    model_inputs['labels'] = labels['input_ids']
    model_inputs['decoder_attention_mask'] = labels['attention_mask']
    return model_inputs

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)

# define training arguments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(device)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8  # Optional but recommended for GPU efficiency
)

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=15,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=500,
    logging_dir='./logs',
    logging_steps=10,
    fp16=False,
)

# define trainer object

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator,
)

# fine tune the model

trainer.train()

# save the model

model.save_pretrained('./HEAL_bart')
tokenizer.save_pretrained('./HEAL_bart')