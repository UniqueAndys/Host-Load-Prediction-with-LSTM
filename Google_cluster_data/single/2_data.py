# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 22:01:34 2016

@author: tyrion

deal with each machine and accumulate the cpu load
"""

from os import path
project_path = '/home/tyrion/lannister/clusterdata-2011-2/python'

from pandas import read_csv
import os
import matplotlib.pyplot as plt
import pickle

rootDir = path.join(project_path,'machine_ids')
list_dirs = os.walk(rootDir)
num = 0
#for lists in os.listdir(rootDir):
for root, dirs, files in list_dirs:      
    for f in files:
        num += 1
        print num
        machine_id = f.split('.')[0]
        lists = path.join(root, f)
        print lists
        single_machine_cpu_usage_df = read_csv(lists,compression='gzip')
        f_machine_cpu_usage = []
        for i in xrange(8352):
    #        print i
            period_df = single_machine_cpu_usage_df[(single_machine_cpu_usage_df.start_time>=(600+i*300))&
                                                    (single_machine_cpu_usage_df.start_time<(900+i*300))]
            single_sum = 0
            for j in period_df.index:
                single_sum += (period_df.loc[j,'end_time'] - period_df.loc[j,'start_time'])*period_df.loc[j,'cpu_usage']/300
            f_machine_cpu_usage.append(single_sum)
    #    plt.figure()
    #    plt.plot(f_machine_cpu_usage)
        with open(path.join(project_path,'data','all',machine_id+'.pkl'),'wb') as f:
            pickle.dump(f_machine_cpu_usage, f)
