# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 21:42:06 2016

@author: tyrion
"""

from os import path
project_path = '/home/tyrion/lannister/clusterdata-2011-2/python'

import matplotlib.pyplot as plt
import pickle

machine_id = 563849022

input_id = open(path.join(project_path,'data','563849022.pkl'),'rb')
a = pickle.load(input_id)
plt.plot(a)
plt.xlabel("Time(Day)")
plt.ylabel("CPU Rate")
plt.xlim([0,8352])
plt.xticks([d*12*24 for d in range(3,30,3)],['%i'%d for d in range(3,30,3)])
plt.ylim([0,0.6])
plt.grid(True)
#plt.text(5200,0.06,r'machine_id:563849022',color='red')
plt.savefig("host_load_29d.png", dpi=300, format='png')
plt.show()