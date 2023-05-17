import pandas as pd
import os

original_path = './outputs_new/' #flan-t5-base_0_shot_cot_incontext.csv
files_path = []

for dirpath, dnames, fnames in os.walk(original_path):
    for f in fnames:
        if f.endswith(".csv"):
            files_path.append(os.path.join(dirpath, f))


models = [ 'base', 'small', 'large', 'xl', 'xxl', 'davinci' ]
#compared_models = [ [models[i], models[j]] for j in range(1,len(models)) for i in range(j) ]
compared_models = [
                    #['base', 'small'], ['base', 'large'], ['small', 'large'], 
                    ['base', 'xl'], 
                    #['small', 'xl'], ['large', 'xl'], ['base', 'xxl'], ['small', 'xxl'], 
                    #['large', 'xxl'], ['xl', 'xxl'], ['base', 'davinci'],
                    #['small', 'davinci'], ['large', 'davinci'], ['xl', 'davinci'], ['xxl', 'davinci']
                    ]

to_compare = []

for model_pairs in compared_models:
    _small, _large = model_pairs[0], model_pairs[1]
    shots = [str(i)+'_shot' for i in range(6)]
    run_type = [ 'normal', 'cot' ]
    heuristic = ['incontext', 'random', 'top5']
    
    for _type in run_type:
        for _heuristic in heuristic:
            for shot in shots:
                small_file = ''
                large_file = ''
                for path in files_path:
                    if _type in path and shot in path and _heuristic in path:
                        if _small in path:
                            small_file = path
                        elif _large in path:
                            large_file = path
                if(len(small_file) and len(large_file)):
                    to_compare.append([[small_file, _small], [large_file, _large]])


for files in to_compare:
    qualitative_analysis_list = []
    small_df = pd.read_csv(files[0][0])
    large_df = pd.read_csv(files[1][0])
    try:
        for ind in range(len(small_df)):
            small_row = dict(small_df.iloc[ind])
            large_row = dict(large_df.iloc[ind])
            if small_row['correct'] is True and large_row['correct'] is False:
                row = {}
                row['small_model'] = files[0][1]
                row['large_model'] = files[1][1]
                row['left'], row['pron'], row['right'] = small_row['left'], small_row['pron'], small_row['right']
                row['sentence'] = small_row['left'] + '' +  small_row['pron'] + '' + small_row['right']
                row['small_model_output'], row['large_model_output'] = small_row['predicted'], large_row['predicted']
                row['correct'] = small_row['correct_sent']
                print(row['sentence'])
                qualitative_analysis_list.append(row)     
        if(len(qualitative_analysis_list)):
            #print(files[0][1], ' and ', files[1][1])
            final_df = pd.DataFrame(qualitative_analysis_list)
            #print(final_df)
            final_df.to_csv(f"{original_path}qualitative_analysis_{files[0][1]}_{files[1][1]}.csv")
            print("Dataframe successfully saved!!!")       
    except:
        print("Both dataframes doesn't have same rows. Fix it !!!")