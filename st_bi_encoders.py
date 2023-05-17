#!/usr/bin/env python
"""\
Authors: Jay Sinha

Usage: python3 st_bi_encoders.py
"""

from torch.utils.data import DataLoader
import torch
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryClassificationEvaluator
import logging
from datetime import datetime
import os
import gzip
import csv
import pandas as pd
import wandb
import datetime
import os.path
from os import path
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
wandb.login()

folder_prefix = './'

today = datetime.date.today()
today_date = today.strftime('%Y%m%d')



if path.exists(folder_prefix + today_date) == False:
  os.mkdir(folder_prefix + today_date)

output_folder = folder_prefix + today_date + '/'
processed_path = folder_prefix + 'processed/'

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
dpr_dataset_path = processed_path + 'dpr_data.csv'


# Load a pre-trained sentence transformer model

model_name = 'stsb-roberta-large'
# model_name = 'sentence-t5-large'
# model_name = 'all-mpnet-base-v2'


# Read the dataset
from datetime import datetime

train_batch_size = 16
num_epochs = 8
weight_decay = 0.01
warmup = 0.1

wandb.init(
    # set the wandb project where this run will be logged
    project="cs685-project",
    entity="large-lm",
    # track hyperparameters and run metadata
    config={
    "architecture": "bi-encoder-" + model_name,
    "dataset": "DPR",
    "epochs": num_epochs,
    "train_batch_size": train_batch_size,
    "weight_decay": weight_decay,
    "evaluator": "BinaryClassificationEvaluator",
    "loss": "OnlineContrastiveLoss"
    }
)

model = SentenceTransformer(model_name).to(device)

model_save_path = './' + model_name + '_b' + str(train_batch_size) + '_e' + str(num_epochs) + '_training_be_dpr'+ '-' +datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

dpr_data = pd.read_csv(processed_path + 'DPR Fixed Set.tsv', sep='\t')

# Convert the dataset to a DataLoader ready for training
logging.info("Read DPR train dataset")

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
        score = 1.0
      else:
        score = 0.0
      inp_example = InputExample(texts=[sentence1, sentence2], label=score)
      if row['split'] == 'dev':
          dev_samples.append(inp_example)
      elif row['split'] == 'test':
          test_samples.append(inp_example)
      else:
          train_samples.append(inp_example)



train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.OnlineContrastiveLoss(model=model)



# Development set: Measure correlation between cosine score and gold labels
logging.info("Read DPR dev dataset")
evaluator = BinaryClassificationEvaluator.from_input_examples(dev_samples, name='dpr-dev')

# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * warmup) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          weight_decay=weight_decay)

wsc_data_path =  folder_prefix + 'wsc273.tsv'
wsc_data = pd.read_csv(wsc_data_path, sep='\t')

model.eval()

wsc_results = []
c = 0
for _, row in wsc_data.iterrows():
  row = dict(row)
  cand = row['candidates'].split(',')
  scores = []
  for j in range(2):
    # score = model.predict((row['left'] + ' ' + row['pron'] + ' ' + row['right'], row['left'] + ' ' + cand[j] + ' ' + row['right']))
    embedding1 = model.encode(row['left'] + ' ' + row['pron'] + ' ' + row['right'])
    embedding2 = model.encode(row['left'] + ' ' + cand[j] + ' ' + row['right'])
    score = util.cos_sim(embedding1, embedding2).tolist()[0][0]
    scores.append(score)
  max_index = scores.index(max(scores))
  row['be_score_0'] = scores[0]
  row['be_score_1'] = scores[1]
  if max_index == row['selected']:
    row['correct'] = True
    c+=1
  else:
    row['correct'] = False
  wsc_results.append(row)
# wandb.log({'wsc_fixed_test':(c/285)*100 })

# DPR Train Set Performance

dpr_train_results = []
c = 0
for _, row in dpr_data.iterrows():
  
  row = dict(row)
  if row['split'] == 'train':
    cand = row['candidates'].split(',')
    left = row['source'].split(row['pronoun'])[0]
    right = row['source'].split(row['pronoun'])[1]
    scores = []
    for j in range(2):
        # score = model.predict((row['source'], left + ' ' + cand[j] + ' ' + right))
        embedding1 = model.encode(row['source'])
        embedding2 = model.encode(left + ' ' + cand[j] + ' ' + right)
        score = util.cos_sim(embedding1, embedding2).tolist()[0][0]
        scores.append(score)
    max_index = scores.index(max(scores))
    row['be_score_0'] = scores[0]
    row['be_score_1'] = scores[1]
    if cand[0] == row['target']:
        correct = 0
    else:
        correct = 1
    row['selected'] = correct
    if max_index == correct:
        row['correct'] = True
        c+=1
    else:
        row['correct'] = False
    dpr_train_results.append(row)

wandb.log({'dpr_train_fixed_test':(c/len(dpr_train_results))*100 })
print("Fine Tuned Bi-Encoder: " + model_name + " DPR Train Set Performance: ", (c/len(dpr_train_results))*100)

# DPR Dev Set Performance
dpr_dev_results = []
c = 0
for _, row in dpr_data.iterrows():
  row = dict(row)
  if row['split'] == 'dev':
    cand = row['candidates'].split(',')
    left = row['source'].split(row['pronoun'])[0]
    right = row['source'].split(row['pronoun'])[1]
    scores = []
    for j in range(2):
        embedding1 = model.encode(row['source'])
        embedding2 = model.encode(left + ' ' + cand[j] + ' ' + right)
        score = util.cos_sim(embedding1, embedding2).tolist()[0][0]
        scores.append(score)
    max_index = scores.index(max(scores))
    row['be_score_0'] = scores[0]
    row['be_score_1'] = scores[1]
    if cand[0] == row['target']:
        correct = 0
    else:
        correct = 1
    row['selected'] = correct
    if max_index == correct:
        row['correct'] = True
        c+=1
    else:
        row['correct'] = False
    dpr_dev_results.append(row)
wandb.log({'dpr_dev_fixed_test':(c/len(dpr_dev_results))*100 })
print("Fine Tuned Bi-Encoder: " + model_name + " DPR Dev Set Performance: ", (c/len(dpr_dev_results))*100)
