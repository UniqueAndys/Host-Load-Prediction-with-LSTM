import matplotlib.pyplot as plt

LSTM = [.000166, .000234, .000304, .000353, .000397, .000439]
ESN = [.000197, .000291, .000395, .000481, .000572, .000659]

plt.figure()
plt.plot(ESN, 'rs--', label="ESN")
plt.plot(LSTM, 'bo-', label="Our Method")

plt.xlabel("Prediction Length")
plt.ylabel("MSE")
plt.xlim([-0.5, 5.5])
plt.ylim([.00013, .00070])
plt.xticks([d for d in range(6)],['%i '% ((d+1)*6) for d in range(6)])
plt.grid(True)
plt.legend(loc=4)
plt.savefig("mse_axp0.png", dpi=300, format='png')
plt.show()



LSTM = [.000331, .000410, .000413, .000458, .000529, .000560]
ESN = [.000474	, .000561, .000559, .000640, .000682, .000684]

plt.figure()
plt.plot(ESN, 'rs--', label="ESN")
plt.plot(LSTM, 'bo-', label="Our Method")

plt.xlabel("Prediction Length")
plt.ylabel("MSE")
plt.xlim([-0.5, 5.5])
plt.ylim([.00030, .00070])
plt.xticks([d for d in range(6)],['%i '% ((d+1)*6) for d in range(6)])
plt.grid(True)
plt.legend(loc=4)
plt.savefig("mse_axp7.png", dpi=300, format='png')
plt.show()



LSTM = [.000581, .000750, .000861, .000976, .001057, .001164]
ESN = [.000780	, .000935	, .001014	, .001129	, .001245	, .001319]

plt.figure()
plt.plot(ESN, 'rs--', label="ESN")
plt.plot(LSTM, 'bo-', label="Our Method")

plt.xlabel("Prediction Length")
plt.ylabel("MSE")
plt.xlim([-0.5, 5.5])
plt.ylim([.00055, .00135])
plt.xticks([d for d in range(6)],['%i '% ((d+1)*6) for d in range(6)])
plt.grid(True)
plt.legend(loc=4)
plt.savefig("mse_sahara.png", dpi=300, format='png')
plt.show()



LSTM_1 = [.000530, .000697, .000851, .000971, .001058, .001137]
LSTM_0 = [.000618, .000807, .001017, .001174, .001375, .001362]
ESN = [.000789, .000874, .001117, .001381, .001557, .001663]

plt.figure()
plt.plot(ESN, 'rs--', label="ESN")
plt.plot(LSTM_0, 'gv-.', label="Our method before training")
plt.plot(LSTM_1, 'bo-', label="Our method after training")

plt.xlabel("Prediction Length")
plt.ylabel("MSE")
plt.xlim([-0.5, 5.5])
plt.ylim([.00050, .0017])
plt.xticks([d for d in range(6)],['%i '% ((d+1)*6) for d in range(6)])
plt.grid(True)
plt.legend(loc=4)
plt.savefig("mse_themis.png", dpi=300, format='png')
plt.show()
