# word2vec only
import random
import time
import os
import math
import codecs
import copy
import numpy as np
from gensim.models import word2vec

file = "/home/aistudio/"
entity = file + "entity_with_text.txt"
relation = file + "relation_with_text.txt"
train = file + "train.txt"
test = file + "test.txt"
test_output = file + "test_output.txt"
corpus = file + "corpus.txt"

with open(entity, 'r') as fr1, open(relation, 'r') as fr2, open(corpus, 'w') as fw: 
    for line in fr1.readlines():
        fw.write(line.strip().split('\t')[1])
        fw.write("\n")

    for line in fr2.readlines():
        fw.write(line.strip().split('\t')[1])
        fw.write("\n")

sentences = word2vec.LineSentence(corpus)
model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=3, vector_size=200)

entity_vec = dict()    # entity id <-> vector of entity 
relation_vec = dict()
with open(entity, 'r') as fr1, open(relation, 'r') as fr2:
    for line in fr1.readlines():
        entity = line.strip().split('\t')[0]
        words = line.strip().split('\t')[1].split(" ")
        entity_vec[entity] = 0
        for word in words:
            entity_vec[entity] += model.wv[word]    
        entity_vec[entity] = entity_vec[entity] / np.linalg.norm(entity_vec[entity])
    
    for line in fr2.readlines():
        relation = line.strip().split('\t')[0]
        words = line.strip().split('\t')[1].split(" ")
        relation_vec[relation] = 0
        for word in words:
            relation_vec[relation] += model.wv[word]    
        relation_vec[relation] = relation_vec[relation] / np.linalg.norm(relation_vec[relation])

with open(test, 'r') as fr, open(test_output, 'w') as fw: 
    testnum = 0
    begin = True       
    for line in fr.readlines(): 
        if begin == True:
            begin = False
        else:
            fw.write("\n")

        testnum += 1
        entity = line.strip().split('\t')[0]
        relation = line.strip().split('\t')[1]
        if entity not in entity_vec:
            fw.write("1,2,3,4,5")
            continue

        distance = dict()
        e1 = entity_vec[entity]
        r = relation_vec[relation]
        for other_entity in entity_vec:
            e2 = entity_vec[other_entity]
            distance[other_entity] = np.dot(e2, r) + np.dot(e1, e2)     # result4
            # distance[other_entity] = np.dot(e1, e2)  -- result3
            # distance[other_entity] = np.dot(e2, r) * np.dot(e1, e2) -- result5
            # Other parameters/expressions?

        SortedList = sorted(distance.items(), key = lambda x:x[1], reverse=True)
        for i in range(5):
            if i < 4:
                fw.write(str(SortedList[i][0])+",")
            else:
                fw.write(str(SortedList[i][0]))
        if testnum % 1000 == 0:
            print(testnum)