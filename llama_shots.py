#!/usr/bin/env python
"""\
Authors: Jay Sinha, Kunal Kumar

Usage: python3 llama_shots.py
"""
import torch
import pandas as pd
import warnings
import pandas as pd
import torch
import csv
import re
import os
import argparse
import logging
from pathlib import Path
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from typing import Tuple
import sys
import time
import json
"""Example:

Jane knocked on Susan’s door, but there was no answer.
OPTIONS:
- Jane was out.
- Susan was out
"""

warnings.filterwarnings('ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



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

def run_model_test(given_path, models):
    wsc_data_path = given_path + 'wsc273.tsv'
    wsc_data = pd.read_csv(wsc_data_path, sep='\t')
    allIncontextExampleCsv = pd.read_csv(f"{given_path}allIncontextExampleUpdated.csv")
    final_result = {}
    allIncontextExampleDict = {}
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")
    temperature = 0.8
    top_p = 0.95
    max_seq_len = 512
    max_batch_size = 32
    for _,row in allIncontextExampleCsv.iterrows():
        row = dict(row)
        allIncontextExampleDict[row['test_sentence']] = row['score_wise_sentence']

    for key, value in allIncontextExampleDict.items():
        value = value.replace('[[','').replace(']]','')
        temp_val = value.split('], [')
        tempVal_list = [ re.sub('\d', '', val) for val in temp_val ]
        allIncontextExampleDict[key] = tempVal_list

    for i in models.keys():
        ckpt_dir = models[i]['model_path']
        tokenizer_path = models[i]['tokenizer_path']
        generator = load(ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size)

        scores = generateOutputsFewShot(i, generator, wsc_data, allIncontextExampleDict)
        final_result[i] = scores
        break
    print(final_result)


def generateOutputsFewShot(model_name, generator, wsc_data, 
                           allIncontextExampleDict, shots = 6, temperature: float = 0.8, 
                           top_p: float = 0.95):
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
            
            prompts = [ str(value).replace('Question => ','').replace("'","").replace('"','').replace("            "," ").replace("Output =>",os.linesep+"Output :") for value in key_sentence_incontext_eg[:shot] ]
            prompts = [ "The Output of below sentence is : " + os.linesep +prompt[3:]+ os.linesep + os.linesep for prompt in prompts ]

            prompt_string = "".join(prompts)
            input_sentence = prompt_string + 'What is the Output of below sentence ?'+ os.linesep + row['left'] + ' OPTIONS: - ' + cand[0] + ' ' + row['right'] +' - ' + cand[1] + ' ' +  row['right']
            input_sentence += os.linesep + "Output : "
            row['input_sentence'] = input_sentence

            prompts = [
            input_sentence
            ]
            row['predicted'] = generator.generate(
                prompts, max_gen_len=256, temperature=temperature, top_p=top_p
            )
            
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

        with open('outputs/' +model_name + '_' + str(shot) + '_shot' + '.csv', 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(wsc_results)
        print(model_name, shot, (c/len(wsc_data))*100 )
        result.append((c/len(wsc_data))*100)
        break
    return result

    
def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


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
        'llama-7b': {'model_path': '/work/pi_mccallum_umass_edu/jsinha_umass_edu/llama/llama_model',
                      'tokenizer_path': '/work/pi_mccallum_umass_edu/jsinha_umass_edu/llama/tokenizer'},
        'llama-13b': {'model_path': '/work/pi_mccallum_umass_edu/jsinha_umass_edu/llama/llama_13b',
                      'tokenizer_path': '/work/pi_mccallum_umass_edu/jsinha_umass_edu/llama/tokenizer'},
    }
    run_model_test(given_path, models)


# def main(
#     ckpt_dir: str,
#     tokenizer_path: str,
#     temperature: float = 0.8,
#     top_p: float = 0.95,
#     max_seq_len: int = 512,
#     max_batch_size: int = 32,
# ):
#     local_rank, world_size = setup_model_parallel()
#     if local_rank > 0:
#         sys.stdout = open(os.devnull, "w")

#     generator = load(
#         ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
#     )


#     wsc_results = []
#     c = 0
#     for _, row in wsc_data.iterrows():
#         row = dict(row)
#         cand = row['candidates'].split(',')
#         scores = []
#         correct = cand[row['selected']] + ' ' + row['right']
#         input_sentence = 'Input: "The bee landed on the flower because ' + '\
#         Options: ' + '\
#             - The bee had pollen.' + '\
#             - the flower had pollen.' + '\
#         Answer: the flower had pollen.' + '\
#         ###' + '\
#         Input: "When Debbie splashed Tina, "' + '\
#         Options: ' + '\
#             - Debbie got in trouble.' + '\
#             - Tina got in trouble.' + '\
#         Answer: Debbie got in trouble.' + '\
#         ###' + '\
#         Input: "The man stole the neighbor\'s bike because"' + '\
#         Options: ' + '\
#             - The man needed one.' + '\
#             - the neighbor needed one.' + '\
#         Answer: The man needed one.' + '\
#         ###' + '\
#         Input: ' + row['left'] + '\
#         Options:' + '\
#         - ' + cand[0] + ' ' + row['right'] +' \
#         - ' + cand[1] + ' ' +  row['right'] + '\
#         Answer: '

        
#         answers = results[0].split('Answer:')
#         print(answers)
#         answer = answers[4].split('.')
#         print(answer)
#         print(correct)
#         row['predicted'] = answer[0] + '.'
#         row['correct_sent'] = correct
#         if correct in row['predicted']:
#             row['correct'] = True
#             c+=1
#         else:
#             row['correct'] = False
#         wsc_results.append(row)
#     print("LLaMA 7B Few Shot: "  + " WSC Test Set Performance: ", (c/285)*100)
#     fields = list(wsc_results[0].keys())
#     with open('wsc_llama_7b_few_shot.csv', 'w', newline='') as output_file:
#         dict_writer = csv.DictWriter(output_file, fields)
#         dict_writer.writeheader()
#         dict_writer.writerows(wsc_results)