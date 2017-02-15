load = []
with open("./Load_Data/themis.nectar.cs.cmu.edu_Aug_97.asciitrace", 'rb') as f:
    for line in f:
        load.append(float(line.split()[1]))
#print load

max_load = max(load)
min_load = min(load)
regulized_load = []
for i in load:
    regulized_value = 0.1 +(i - min_load)*(0.9-0.1)/(max_load-min_load)
    regulized_load.append(regulized_value)
    
import matplotlib.pyplot as plt
plt.plot(regulized_load)
plt.figure()
plt.plot(regulized_load[:200])

import pickle
with open("./data/themis.pkl", 'wb') as f:
    pickle.dump(regulized_load, f)