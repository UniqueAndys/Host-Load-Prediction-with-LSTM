# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 17:14:25 2016

@author: tyrion

sample 100 machines rawly and save it separately
"""

import pickle
import pandas as pd
from pandas import read_csv
from os import path

project_path = '/home/tyrion/lannister/clusterdata-2011-2/python'

input_id = open(path.join(project_path,'info','machine_id.pkl'), 'rb')
all_machine_id = pickle.load(input_id)
input_id.close()

from random import sample
sample_machine_ids = sample(all_machine_id, 100)
print sample_machine_ids

import time
tStart = time.time()
for single_machine_id in sample_machine_ids:
    single_machine_cpu_usage_df = pd.DataFrame()    
    for i in xrange(5):  
        machine_cpu_usage_df = read_csv(path.join(project_path,'info','314','all_machine_cpu_usage_'+
                               str(i)+'.csv.gz'),compression='gzip')
        
        single_machine = machine_cpu_usage_df[machine_cpu_usage_df['machine_id']==single_machine_id]
        single_machine = single_machine.drop(['machine_id'], axis=1)
        single_machine = single_machine[(single_machine.cpu_usage != 0)&(single_machine.cpu_usage <= 1)]    
        single_machine.start_time = single_machine.start_time / 1000000
        single_machine.end_time = single_machine.end_time / 1000000
       
        single_machine_cpu_usage_df = single_machine_cpu_usage_df.append(single_machine, ignore_index=True)
        print i,len(single_machine_cpu_usage_df)
    single_machine_cpu_usage_df.to_csv(path.join(project_path,'machine_ids','329',
                                    str(single_machine_id)+'.csv.gz'),compression='gzip',index=False)
    tEnd = time.time()
    print "It costs %f sec" % (tEnd - tStart)
    print single_machine_cpu_usage_df
