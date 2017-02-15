# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 20:20:30 2016

@author: tyrion

merge the separate machine value into a 1024 large matrix
"""

from os import path
import pickle
import os

project_path = '/home/tyrion/lannister/clusterdata-2011-2/python'
data = []
rootDir = path.join(project_path,'data','all')
list_dirs = os.walk(rootDir)
num = 0
for lists in os.listdir(rootDir):
    lists = path.join(rootDir, lists)
    num += 1
    print num,lists
    input_machine = open(lists,'rb')
    cpu_load = pickle.load(input_machine)
    data.append(cpu_load)
    input_machine.close()

with open(path.join(project_path,'data','tyrion.pkl'),'wb') as f:
    pickle.dump(data, f)