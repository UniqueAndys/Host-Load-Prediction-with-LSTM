# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 21:44:07 2016

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
plt.xlabel("Time(Hour)")
plt.ylabel("CPU Rate")
plt.xlim([0,12*6])
plt.xticks([d*12 for d in range(1,7)],['%ih '%d for d in range(1,7)])
plt.ylim([0.15,0.35])
plt.grid(True)
#plt.text(5200,0.06,r'machine_id:563849022')
plt.savefig("host_load_6h.png", dpi=300, format='png')
plt.show()