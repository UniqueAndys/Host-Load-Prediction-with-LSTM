# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 11:17:33 2016

@author: tyrion

extra 4 column from the original data
save it into 5 csv file
"""

from os import chdir
chdir('/home/tyrion/lannister/clusterdata-2011-2')

import pandas as pd
from pandas import read_csv
from os import path

project_path = '/home/tyrion/lannister/clusterdata-2011-2/python'

task_usage_csv_colnames = ['start_time','end_time','job_id','task_index','machine_id','cpu_usage',
            'can_mem_usage','assi_mem_usage','ummap_page_cache','total_page_cache','max_mem_usage',
            'disk_io_time','disk_space_usage','max_cpu_rate','max_disk_io_time','cyc_per_instr',
            'mem_acc_per_inst','sample_portion','aggre_type','sampled_cpu_usage']

import time

for i in xrange(5):
    tStart = time.time()
    all_cpu_usage_df = pd.DataFrame()    
    for j in xrange(i*100, (i+1)*100):
        if j < 10:
            sub_csv = "0000"+str(j)
        elif j < 100:
            sub_csv = "000"+str(j)
        else:
            sub_csv = "00"+str(j)
            
        task_usage_df = read_csv(path.join('task_usage','part-'+sub_csv+'-of-00500.csv.gz'),header=None,
                        index_col=False,compression='gzip',names=task_usage_csv_colnames)
        
        machine_cpu_load = task_usage_df[['start_time','end_time','machine_id','cpu_usage']]
        all_cpu_usage_df = all_cpu_usage_df.append(machine_cpu_load, ignore_index=True)
        print j,len(all_cpu_usage_df)
    all_cpu_usage_df.to_csv(path.join(project_path,'info','machine','all_cpu_usage_df'+str(i)+'.csv.gz'),compression='gzip',index=False)
    tEnd = time.time()
    print "It costs %f sec" % (tEnd - tStart)
