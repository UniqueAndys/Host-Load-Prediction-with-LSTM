import pickle

with open("./data/axp7.pkl", 'rb') as f:
    axp7 = pickle.load(f)
    
import matplotlib.pyplot as plt

#plt.figure()
#plt.plot(axp7)
#plt.xlabel("measurement")
#plt.ylabel("CPU Rate")
#plt.savefig("axp7.png", dpi=300, format='png')
#plt.show()
#
#plt.figure()
#plt.plot(axp7[100:300])
#plt.xlabel("measurement")
#plt.ylabel("CPU Rate")
#plt.savefig("axp7_detial.png", dpi=300, format='png')
#plt.show()

from numpy import fft
import numpy as np

def plot_fft(time_domain):
    f = fft.fft(time_domain)
    g = np.abs(f)
    print g
    plt.plot(g)
    plt.xlim([0, 50])
#    plt.xticks([d*2 for d in range(25)])
#    plt.ylim([0, 200])
#    plt.grid(True)
    plt.show()
    return g

g = plot_fft(axp7)