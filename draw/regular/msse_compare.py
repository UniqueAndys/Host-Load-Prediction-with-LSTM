import numpy as np

interval_esn = [.002005, .001818, .001927, .002106, .002147, .002047, .002521, .002722]
interval_lstm = [.002012, .001821, .001882, .002003, .002050, .001983, .002453, .002613]

def seg_pattern(seg_hostload):
    seg_pattern = []
    for i in range(len(seg_hostload)):
        if i == 0:
            seg_pattern.append(seg_hostload[0])
        else:
            seg_pattern.append(2*seg_hostload[i] - seg_hostload[i-1])
    return seg_pattern

def cal_msse(seg_pattern):
    msse_all = []
    for j in range(len(seg_pattern)):
        seg_pattern_j = seg_pattern[:j+1]
        s = []
        for i in range(len(seg_pattern_j)):
            if i == 0:
                s.append(2**0)
            else:
                s.append(2 ** (i-1))
        s_sum = np.sum(s)
        msse = 0
        for i in range(len(seg_pattern_j)):
            msse += s[i] * seg_pattern_j[i]
        msse /= s_sum
        msse_all.append(msse)
    return msse_all

seg_pattern_esn = seg_pattern(interval_esn)
seg_pattern_lstm = seg_pattern(interval_lstm)

msse_esn = cal_msse(seg_pattern_esn)
msse_lstm = cal_msse(seg_pattern_lstm)
print msse_esn
print msse_lstm

import matplotlib.pyplot as plt

plt.plot(msse_esn[3:], 'rs--', label="ESN")
plt.plot(msse_lstm[3:], 'bo-', label="Our Method")

plt.xlabel("Prediction Length")
plt.ylabel("MSSE")
plt.xlim([-0.5, 4.5])
plt.ylim([.0017, .0028])
plt.xticks([d for d in range(5)],["0.7h", "1.3h", "2.7h", "5.3h", "10.7h"])
plt.grid(True)
plt.legend(loc=4)
plt.savefig("msse_comparison.png", dpi=300, format='png')
plt.show()
