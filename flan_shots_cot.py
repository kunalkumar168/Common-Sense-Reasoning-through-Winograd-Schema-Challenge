#!/usr/bin/env python
"""\
Authors: Kunal Kumar, Jay Sinha

Usage: python3 flan_shots_cot.py
"""
import torch
import pandas as pd
import warnings
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import torch
import csv
import re
import os
import argparse
import logging
import json
"""Example:

Jane knocked on Susan’s door, but there was no answer.
OPTIONS:
- Jane was out.
- Susan was out
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')


def replace_string(word, replacement):
    to_replace = word
    cleaned_word = re.sub(r"[^A-Za-z0-9]+",'', word)
    if len(cleaned_word)==len(replacement):
        to_replace =  word.replace(replacement, '**')
    return to_replace

def exactMatch(pred, truth, row):
    replacement = re.sub(r'[^A-Za-z0-9 ]+', '', row['right'])
    pred = re.sub(r'[^A-Za-z0-9 ]+', '', pred)
    truth = re.sub(r'[^A-Za-z0-9 ]+', '', truth)
    pred = pred.lstrip()
    pred = pred.rstrip()
    truth = truth.lstrip()
    truth = truth.rstrip()
    truth = truth.lower()
    pred = pred.lower()
    temp_pred = pred.replace(replacement,'')
    temp_truth = truth.replace(replacement,'')
    temp_pred = temp_pred.lstrip()
    temp_pred = temp_pred.rstrip()
    temp_truth = temp_truth.lstrip()
    temp_truth = temp_truth.rstrip()
    row['correct_sent'] = truth
    row['predicted'] = pred
    return temp_pred==temp_truth

def run_model_test(given_path, models, run_type):
    wsc_data_path = given_path + 'wsc273.tsv'
    wsc_data = pd.read_csv(wsc_data_path, sep='\t')
    incontext_files = ['allIncontextExampleNew.csv', 'allIncontextExampleRandom.csv']
    heuristics = ['incontext', 'random']
    h = 0
    for x in incontext_files:
    
        allIncontextExampleCsv = pd.read_csv(given_path + x)
        final_result = {}
        allIncontextExampleDict = {}

        for _,row in allIncontextExampleCsv.iterrows():
            row = dict(row)
            allIncontextExampleDict[row['test_sentence']] = row['score_wise_sentence']

        for key, value in allIncontextExampleDict.items():
            value = value.split('[SEP]')
            value = [ val.strip() for val in value ]
            temp_val = []
            for val in value:
                if(len(val)):
                    temp = val.split('[OUT]')
                    temp_val.append([temp[0].strip(), temp[1].strip()])
                    allIncontextExampleDict[key] = temp_val

        for i in models.keys():
            tokenizer = T5Tokenizer.from_pretrained(models[i])
            model = T5ForConditionalGeneration.from_pretrained(models[i], 
                                                            device_map="auto"
                                                            )
            scores = generateOutputsFewShot(i, model, tokenizer, wsc_data, allIncontextExampleDict, run_type, heuristics[h])
            final_result[i] = scores
        print(x, final_result)
        h = 1

def generateOutputsFewShot(model_name, model, tokenizer, wsc_data, allIncontextExampleDict, run_type, heuristic, shots = 6):
    result = []
    
    for shot in range(shots):
        wsc_results = []
        c = 0
        print(f"Shot undergoing = {shot}")
        for _,row in wsc_data.iterrows():
            row = dict(row)
            cand = row['candidates'].split(',')
            correct = cand[row['selected']] + ' ' + row['right']

            key_sentence = row['left']+' '+row['pron']+' '+row['right']
            key_sentence_incontext_eg = allIncontextExampleDict[key_sentence]
            
            if run_type == 'normal':
              
              prompts = []
              for val in key_sentence_incontext_eg[:shot]:
                left = val[0].split('OPTIONS: ')[0]
                right = val[0].split('OPTIONS: ')[1]
                options = right.split('-')
                prompts.append(
                        'How does the sentence end?' +
                        os.linesep + os.linesep +
                        left +
                        os.linesep + os.linesep +
                        'OPTIONS: ' +
                        os.linesep +
                        '- ' + options[1] + 
                        os.linesep +
                        '- ' + options[2] +
                        os.linesep +
                        'Output: ' + val[1] +
                        os.linesep + 
                        "=======================================" + 
                        os.linesep)
              prompt_string = "".join(prompts)
              input_sentence = prompt_string + 'How does the sentence end?' + os.linesep + os.linesep + row['left'] + os.linesep + os.linesep + ' OPTIONS: ' + os.linesep + '- ' + cand[0] + ' ' + row['right'] + os.linesep + ' - ' + cand[1] + ' ' +  row['right']
              input_sentence += os.linesep + "Output: "
            else:
              prompts = []
              prompt_thought = "Thought : Based on the given two options, we selected one of the two option that best fits the pronoun in the above sentence."
              for val in key_sentence_incontext_eg[:shot]:
                left = val[0].split('OPTIONS: ')[0]
                right = val[0].split('OPTIONS: ')[1]
                options = right.split('-')
                prompts.append(
                        'How does the sentence end?' +
                        os.linesep + os.linesep +
                        left +
                        os.linesep + os.linesep +
                        'OPTIONS: ' +
                        os.linesep +
                        '- ' + options[1] + 
                        os.linesep +
                        '- ' + options[2] +
                        os.linesep +
                        'Output: ' + val[1] +
                        os.linesep + 
                        prompt_thought +
                        os.linesep + 
                        "=======================================" + 
                        os.linesep)
              
              prompt_string = "".join(prompts)

              input_sentence = prompt_string + 'How does the sentence end?' + os.linesep + os.linesep + row['left'] + os.linesep + os.linesep + ' OPTIONS: ' + os.linesep + '- ' + cand[0] + ' ' + row['right'] + os.linesep +' - ' + cand[1] + ' ' +  row['right']
              input_sentence += os.linesep + os.linesep+"Output: "
            
            row['input_prompt'] = input_sentence
            input_ids = tokenizer(input_sentence, return_tensors="pt", max_length = 1000).input_ids.to(device)
            outputs = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, max_new_tokens=1000)
            row['predicted'] = tokenizer.decode(outputs[0]).encode('utf-8').decode("utf-8").strip()[6:-4]
            
            row['correct_sent'] = correct
            pred_out = re.sub(re.compile('<.*?>'), '' ,row['predicted'])
            pred_out = pred_out.lstrip()

            if exactMatch(pred_out, correct, row):
                row['correct'] = True
                c+=1
            else:
                row['correct'] = False
            row['number of shots'] = shot
            wsc_results.append(row)
        
        keys = wsc_results[0].keys()
        print(model_name, shot, (c/len(wsc_data))*100 )
        with open('outputs/' + model_name + '_' + str(shot) + '_shot' + '_' + run_type + '_' + heuristic + '.csv', 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(wsc_results)
            
        result.append((c/len(wsc_data))*100)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--given_path",
        type=str,
        nargs='?',
        const=1,
        default="/home/jsinha_umass_edu/cs685/",
        help="Home Directory where wsc, DPR Fixed Set, All In Context Examples files are present",
    )

    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )

    args = parser.parse_args()
    print(args)
    given_path = args.given_path
    models = {
        'flan-t5-small': 'google/flan-t5-small',
        'flan-t5-base': 'google/flan-t5-base',
        'flan-t5-large': 'google/flan-t5-large',
        # 'flan-t5-xl': '/work/pi_mccallum_umass_edu/jsinha_umass_edu/flan-t5-xl',
        # 'flan-t5-xxl': '/work/pi_mccallum_umass_edu/jsinha_umass_edu/flan-t5-xxl'
    }
    for i in ['cot']:
        run_model_test(given_path, models, i)

    
