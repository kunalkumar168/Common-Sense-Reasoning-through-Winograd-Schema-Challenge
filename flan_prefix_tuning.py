#!/usr/bin/env python
"""\
Authors: Ishita Kumar, Jay Sinha

Usage: python3 flan_prefix_tuning.py
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torch
import pandas as pd
from datasets import load_dataset

import re


file_path = '/content/'
dataset_train = load_dataset("csv", data_files=file_path+ "dpr_train.csv")
dataset_dev = load_dataset("csv", data_files=file_path+"dpr_dev.csv")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_length = 128
lr = 1e-2
num_epochs = 6
batch_size = 8

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

def preprocess_function(examples):
    inputs = examples['source']
    targets = examples['target']
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(targets, max_length= 32, padding="max_length", truncation=True, return_tensors="pt")
    labels = labels["input_ids"]
    #labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs

def preprocess(row):
  cand = row['candidates'].split(',')
  print(cand)
  source = row['source']
  reg = re.search(r'(.*[\s.,]*)(' + row['pronoun'] + ')([\s.,]*.*)', source)
  highlighted_source = reg.groups()[0] + '*' + reg.groups()[1] + '*' + reg.groups()[2]
  input_sentence = input_sentence = highlighted_source + '\
    OPTIONS: \
    - ' + cand[0] + ' ' + ' \
    - ' + cand[1]
  row['source'] = input_sentence
  return row

def exactMatch(pred, truth):
    #replacement = re.sub(r'[^A-Za-z0-9 ]+', '', row['right'])
    pred = re.sub(r'[^A-Za-z0-9 ]+', '', pred)
    truth = re.sub(r'[^A-Za-z0-9 ]+', '', truth)
    pred = pred.lstrip()
    pred = pred.rstrip()
    truth = truth.lstrip()
    truth = truth.rstrip()
    truth = truth.lower()
    pred = pred.lower()
    #temp_pred = pred.replace(replacement,'')
    #temp_truth = truth.replace(replacement,'')
    #temp_pred = temp_pred.lstrip()
    #temp_pred = temp_pred.rstrip()
    #temp_truth = temp_truth.lstrip()
    #temp_truth = temp_truth.rstrip()
    #row['correct_sent'] = truth
    #row['predicted'] = pred
    return pred==truth

dataset_preprocessed = dataset_train.map(
    preprocess,
    batched=False,
    remove_columns=['pronoun','candidates'],
    num_proc=1,
)

dataset_preprocessed_dev = dataset_dev.map(
    preprocess,
    batched=False,
    remove_columns=['pronoun','candidates'],
    num_proc=1,
)


train_dataloader = DataLoader(
    processed_datasets['train'], shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
eval_dataloader = DataLoader(processed_datasets_dev['train'], collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)


peft_config = PrefixTuningConfig(peft_type="PREFIX_TUNING",
                                 task_type=TaskType.SEQ_2_SEQ_LM, 
                                 inference_mode=False, 
                                 num_virtual_tokens=8)

#                                  token_dim=768,
#                                  num_transformer_submodules=1,
#                                  num_attention_heads=12,
#                                  num_layers=12,
#                                  encoder_hidden_size=768,
#                                  prefix_projection=True

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)


model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs =  model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")


correct = 0
total = 0
i = 0
incorrect = []
corrected = []
for pred, true in zip(eval_preds, dataset_preprocessed_dev['train']['target']):
    if exactMatch(pred, true):
        correct += 1
        corrected.append((pred,true))
    else:
      incorrect.append((pred,true))
    total += 1
accuracy = correct / total * 100
print(f"{accuracy=} % on the evaluation dataset")
