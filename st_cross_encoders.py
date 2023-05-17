#!/usr/bin/env python
"""\
Authors: Jay Sinha

Usage: python3 st_cross_encoders.py
"""

from torch.utils.data import DataLoader
import torch
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator, CEBinaryClassificationEvaluator
from sentence_transformers import InputExample
from transformers import TrainingArguments, Trainer
import logging
import datetime
import sys
import os
import os.path
from os import path
import pandas as pd
import csv
import wandb
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
folder_prefix = './'

today = datetime.date.today()
today_date = today.strftime('%Y%m%d')

if path.exists(folder_prefix + today_date) == False:
  os.mkdir(folder_prefix + today_date)

output_folder = folder_prefix + today_date + '/'

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
processed_path = folder_prefix + 'processed/'


model_prefix = 'cross-encoder/'
model_name = 'stsb-roberta-large'

#Define our Cross-Encoder Training Specs
from datetime import datetime
train_batch_size = 16
num_epochs = 4
weight_decay = 0.1
warmup = 0.5
model_save_path = output_folder + model_name + '_b' + str(train_batch_size) + '_e' + str(num_epochs) + '_training_ce_dpr-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

model = CrossEncoder(model_prefix + model_name, device = device, num_labels=1)

wandb.init(
    # set the wandb project where this run will be logged
    project="cs685-project",
    entity="large-lm",
    # track hyperparameters and run metadata
    config={
    "architecture": "cross-encoder-" + model_name,
    "dataset": "DPR",
    "epochs": num_epochs,
    "train_batch_size": train_batch_size,
    "weight_decay": weight_decay,
    "symmetric_redundancy": "Yes",
    "evaluator": "CEBinaryClassificationEvaluator",
    "dpr_data": "Fixed",
    "warmup": warmup,
    }
)

dpr_data = pd.read_csv(processed_path + 'DPR Fixed Set.tsv', sep='\t')

# Read DPR dataset
logger.info("Read DPR train dataset")

train_samples = []
dev_samples = []
test_samples = []
for _, row in dpr_data.iterrows():
  row = dict(row)
  cands = row['candidates']
  for i in range(2):
    sentence1 = row['sentence']
    sentence2 = row['left'] + cands[i] + row['right']
    if i == row['label']:
      score = 1
    else:
      score = 0
    # score = float(row['label']) # Normalize score to range 0 ... 1

    if row['split'] == 'dev':
        dev_samples.append(InputExample(texts=[sentence1, sentence2], label=score))
    elif row['split'] == 'test':
        test_samples.append(InputExample(texts=[sentence1, sentence2], label=score))
    else:
        #As we want to get symmetric scores, i.e. CrossEncoder(A,B) = CrossEncoder(B,A), we pass both combinations to the train set
        train_samples.append(InputExample(texts=[sentence1, sentence2], label=score))
        train_samples.append(InputExample(texts=[sentence1, sentence2], label=score))

# We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)


# We add an evaluator, which evaluates the performance during training
evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_samples, name='dpr-dev')

# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * warmup)
logger.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          weight_decay=weight_decay)

model = CrossEncoder(model_save_path, device = device).eval()

wsc_data_path =  folder_prefix + 'wsc273.tsv'
wsc_data = pd.read_csv(wsc_data_path, sep='\t')

wsc_results = []
c = 0
for _, row in wsc_data.iterrows():
  row = dict(row)
  cand = row['candidates'].split(',')
  scores = []
  for j in range(2):
    score = model.predict((row['left'] + ' ' + row['pron'] + ' ' + row['right'], row['left'] + ' ' + cand[j] + ' ' + row['right']))
    scores.append(score)
  max_index = scores.index(max(scores))
  row['ce_score_0'] = scores[0]
  row['ce_score_1'] = scores[1]
  if max_index == row['selected']:
    row['correct'] = True
    c+=1
  else:
    row['correct'] = False
  wsc_results.append(row)
wandb.log({'wsc_fixed_test':(c/285)*100 })
print("Fine Tuned Cross-Encoder: " + model_name + " WSC Test Set Performance: ", (c/285)*100)

# DPR Train Set Performance
dpr_train_results = []
c = 0
for _, row in dpr_data.iterrows():
  
  row = dict(row)
  if row['split'] == 'train':
    cands = row['candidates']
    scores = []
    for i in range(2):
      sentence1 = row['sentence']
      sentence2 = row['left'] + cands[i] + row['right']
      score = model.predict((sentence1, sentence2))
      scores.append(score)
    max_index = scores.index(max(scores))
    row['ce_score_0'] = scores[0]
    row['ce_score_1'] = scores[1]
    if max_index == row['label']:
      row['correct'] = True
      c+=1
    else:
      row['correct'] = False
    dpr_train_results.append(row)
wandb.log({'dpr_train_fixed_test':(c/len(dpr_train_results))*100 })
print("Fine Tuned Cross-Encoder: " + model_name + " DPR Train Set Performance: ", (c/len(dpr_train_results))*100)

# DPR Dev Set Performance
dpr_dev_results = []
c = 0
for _, row in dpr_data.iterrows():
  row = dict(row)
  if row['split'] == 'dev':
    cands = row['candidates']
    scores = []
    for i in range(2):
      sentence1 = row['sentence']
      sentence2 = row['left'] + cands[i] + row['right']
      score = model.predict((sentence1, sentence2))
      scores.append(score)
    max_index = scores.index(max(scores))
    row['ce_score_0'] = scores[0]
    row['ce_score_1'] = scores[1]
    if max_index == row['label']:
      row['correct'] = True
      c+=1
    else:
      row['correct'] = False
    dpr_dev_results.append(row)
wandb.log({'dpr_dev_fixed_test':(c/len(dpr_dev_results))*100 })
print("Fine Tuned Cross-Encoder: " + model_name + " DPR Dev Set Performance: ", (c/len(dpr_dev_results))*100)

wandb.finish()
