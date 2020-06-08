# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 18:18:40 2020

@author: canbe
"""
import numpy as np
import pandas as pd

root = "C:/pr_files/neural/"
with open('glove.6B.300d.txt', 'r', encoding="utf8") as f:
    glove = [[str(s) for s in line.rstrip().split(' ')] for line in f]

words = [g[0] for g in glove]     
#glove = np.array(glove)
#glove[:,300] = np.char.strip(glove[:,300],'\n')
#idx = glove[:,0]
#gloveEmbeds = glove[:,1:].astype(np.float)
wordC = pd.read_hdf(root + "eee443_project_dataset_train.h5", 'word_code')
wordC = wordC.to_dict('split')
wordDict = dict(zip(wordC['data'][0], wordC['columns']))

embeds = []
for i in range(4):
    x = np.random.normal(loc=0, scale=0.02, size=300)
    embeds.append(x)
for i in range(1000):
    word = wordDict[i+4]
    if (word == 'xCatch'):
        word = 'catch'
    if (word == 'xWhile'):
        word = 'while'           
    if (word == 'xCase'):
        word = 'case'  
    if (word == 'xEnd'):
        word = 'end'
    if (word == 'xFor'):
        word = 'for'
    index = words.index(word)
    embed = np.array([float(gem) for gem in glove[index][1:]])
    embeds.append(embed)
embeds = np.array(embeds)
np.savetxt('embeds300.txt', embeds, delimiter=',')