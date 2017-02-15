import numpy as np
import matplotlib.pyplot as plt


n_groups = 3

Bayes = [0.0160, 0.0157, 0.0153]
GMDH = [0.0037, 0.0047, 0.0054]
LSTM = (0.001983, 0.002453, 0.002613)
ESN = (.002047, .002521, .002722)

for i in range(3):
    Bayes[i] /= 3.2
    GMDH[i] /= 1.3

index = np.arange(n_groups)
bar_width = 0.18

opacity = 0.5

rects1 = plt.bar(index, Bayes, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Bayes')

rects2 = plt.bar(index + bar_width, GMDH, bar_width,
                 alpha=opacity,
                 color='g',
                 label='PSR+EA-GMDH')
                 
rects3 = plt.bar(index + bar_width*2, ESN, bar_width,
                 alpha=opacity,
                 color='b',
                 label='ESN')

rects4 = plt.bar(index + bar_width*3, LSTM, bar_width,
                 alpha=opacity,
                 color='y',
                 label='Our method')

plt.xlabel('Prediction Length')
plt.ylabel('Average MSSE')
#plt.title('Scores by group and gender')
plt.xticks(index + bar_width*2, ('2.7h', '5.3h', '10.7h'))
plt.yticks(np.arange(0, 0.008, 0.002))
plt.xlim([-0.2, 4.2])
#plt.legend(bbox_to_anchor=(1.02, 0.7), loc=2)
plt.legend(loc=1)

plt.tight_layout()
plt.savefig("msse.png", dpi=300, format='png')
