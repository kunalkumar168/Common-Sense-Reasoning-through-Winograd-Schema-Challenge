#!/usr/bin/env python
"""\
Authors: Kunal Kumar, Jay Sinha

Usage: python3 plot_results.py
"""


## Plot - 1

import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

def plotResults( large, xl, xxl, davinci, metric ):
    barWidth = 0.15

    fig = plt.subplots(figsize=(10, 8))
    x = range(6)

    # Set position of bar on X axis
    br1 = np.arange(len(large))
    br2 = [x1 + barWidth for x1 in br1]
    br3 = [x1 + barWidth for x1 in br2]
    br4 = [x1 + barWidth for x1 in br3]

    # Make the plot
    plt.bar(br1, large, color ='r', width = barWidth,
            edgecolor ='grey', label ='flant5-large')
    
    plt.bar(br2, xl, color ='b', width = barWidth,
            edgecolor ='grey', label ='flant5-xl')

    plt.bar(br3, xxl, color ='g', width = barWidth,
            edgecolor ='grey', label ='flant5-xxl')
    
    plt.bar(br4, davinci, color ='c', width = barWidth,
            edgecolor ='grey', label ='davinci-text-003')
    
    # Adding Xticks
    #ax.spines[['right', 'top']].set_visible(False)
    plt.xlabel('Number of shots', fontweight ='bold', fontsize = 10)
    plt.ylabel('Accuracy', fontweight ='bold', fontsize = 10)
    plt.xticks([r + barWidth for r in range(len(large))], x)
    
    plt.title(f'Results of FLAN-T5 - for {metric}')
    plt.ylim(60, 100)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon = False)
    #plt.show()
    plt.savefig('exact_match_models.png')

scoreLargeUpdated = [81.75438596491227, 82.80701754385966, 82.80701754385966, 83.15789473684211, 82.80701754385966, 83.85964912280703]
scoreXLUpdated = [87.36842105263159, 89.82456140350877, 89.82456140350877, 88.7719298245614, 89.47368421052632, 89.12280701754386]
scoreXXLUpdated = [89.82456140350877, 89.47368421052632, 91.57894736842105, 90.87719298245615, 90.17543859649123, 90.52631578947368]
scoreDaVinciUpdated = [61.05263157894737, 75.08771929824562, 75.78947368421053, 76.49122807017544, 76.14035087719299, 75.43859649122807]
#plotResults(scoreLargeUpdated, scoreXLUpdated, scoreXXLUpdated, scoreDaVinciUpdated, metric = "Exact Match")


## Plot - 2

import matplotlib as mpl
import numpy as np

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

def plotCompared(_combined, _type ):

    fig, axs = plt.subplots(2, 2, figsize=(15, 15), gridspec_kw={'hspace': 0.4})
    x = range(6)

    # Set position of bar on X axis
    barWidth = 0.15
    _dict = {0:"flant5-large", 1:"flant5-xl", 2:"flant5-xxl", 3:"davinci-txt-003"}

    for i in range(len(_combined)):
      br1 = np.arange(len(_combined[i][0]))
      br2 = [x1 + barWidth for x1 in br1]
      j = 0 if i<2 else 1
      axs[i%2,j].bar(br1, _combined[i][0], color ='r', width = barWidth,
            edgecolor ='grey', label ='normal-prompt')
    
      axs[i%2,j].bar(br2, _combined[i][1], color ='b', width = barWidth,
            edgecolor ='grey', label ='cot-prompt')
      
      axs[i%2,j].set_title(f'For Model : {_dict[i]}', fontsize = 10)
      axs[i%2,j].set_ylim([60, 100])
      axs[i%2,j].legend(fontsize=8, ncol=2)

    plt.tight_layout()
    
    fig.suptitle(f'Plot of {_type} for number of shots')
    for ax in axs.flat:
        ax.set(xlabel="Number of shots", ylabel='Accuracy')
        ax.title.set_color('blue')
        ax.xaxis.label.set_color('green')
        ax.yaxis.label.set_color('red')
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    #plt.show()
    plt.savefig(f'Plot - {_type}.png')

scoreLargeCot = [80.35087719298247, 80.0, 80.0, 80.7017543859649, 80.7017543859649, 80.35087719298247]
scoreXLCot = [88.0701754385965, 90.17543859649123, 90.17543859649123, 89.12280701754386, 88.7719298245614, 89.12280701754386]
scoreXXLCot = [90.17543859649123, 90.17543859649123, 90.52631578947368, 89.47368421052632, 89.47368421052632, 89.47368421052632]
scoreDaVinciCot = [70.87719298245614, 74.3859649122807, 75.9298245614035, 74.03508771929825, 74.3859649122807, 73.68421052631578]

_combined = [[scoreLargeUpdated, scoreLargeCot ], [scoreXLUpdated, scoreXLCot], 
             [scoreXXLUpdated, scoreXXLCot], [scoreDaVinciUpdated, scoreDaVinciCot]]

#plotCompared(_combined, "Normal Prompt Vs Chain of Thought Prompt")