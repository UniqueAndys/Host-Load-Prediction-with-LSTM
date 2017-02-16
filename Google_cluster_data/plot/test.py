# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 21:44:07 2016

@author: tyrion
"""

import matplotlib.pyplot as plt
import pickle

input_id = open("./machine.pkl", 'rb')
a = pickle.load(input_id)
#plt.plot(a[400:600])
#plt.savefig("draw.png", dpi=300, format='png')

start = 400
history = 200
prediction = 50

plt.figure(figsize=(20,3))
plt.plot(a[start:start+history+prediction])
#plt.plot(a[start:start+history], 'b', label="history")
#plt.plot(range(history-1,history+prediction), 
#         a[start+history-1:start+history+prediction], 
#         'r', label="prediction")
plt.ylim([0, 0.35])
plt.savefig("draw.png", dpi=300, format='png')