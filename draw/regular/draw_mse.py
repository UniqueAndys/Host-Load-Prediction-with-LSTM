import matplotlib.pyplot as plt

LSTM = [.003547, .004234, .004773, .005076, .005394, .005665]
ESN = [.003804, .004710, .004991, .005323, .005645, .005957]

plt.plot(ESN, 'rs--', label="ESN")
plt.plot(LSTM, 'bo-', label="Our Method")

plt.xlabel("Prediction Length")
plt.ylabel("MSE")
plt.xlim([-0.5, 5.5])
plt.ylim([.0033, .0062])
plt.xticks([d for d in range(6)],['%i '% ((d+1)*6) for d in range(6)])
plt.grid(True)
plt.legend(loc=4)
plt.savefig("comparison_mse.png", dpi=300, format='png')
plt.show()
