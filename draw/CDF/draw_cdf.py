import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("./data/GMDH.pkl") as input_file:
    GMDH = pickle.load(input_file)
with open("./data/ANN.pkl") as input_file:
    ANN = pickle.load(input_file)
with open("./data/AR.pkl") as input_file:
    AR = pickle.load(input_file)
with open("./data/LSTM.pkl") as input_file:
    LSTM = pickle.load(input_file)
with open("./data/ESN.pkl") as input_file:
    ESN = pickle.load(input_file)

for i in range(4):    
    lstm_sorted_mse = np.sort(LSTM[i])   
    lstm_yvals = np.arange(len(lstm_sorted_mse))/float(len(lstm_sorted_mse))
    esn_sorted_mse = np.sort(ESN[i])   
    esn_yvals = np.arange(len(esn_sorted_mse))/float(len(esn_sorted_mse))
    plt.figure()
    plt.xlabel("MSE Of Prediction")
    plt.ylabel("CDF")
    plt.xlim([0,0.025])
    plt.ylim([0,1])
    predict_length = float(i+1)*0.5
    plt.title("T="+str(predict_length)+"h")
    plt.plot(lstm_sorted_mse, lstm_yvals, 'y-', label="Our Method")
    plt.plot(esn_sorted_mse, esn_yvals, 'c--', label="ESN")
#    plt.plot(rnn_sorted_mse, rnn_yvals, 'k', label="SRN")
    plt.plot(GMDH[i*2], GMDH[i*2+1], 'r-.', label="PSR+EA-GMDH")
    plt.plot(ANN[i*2], ANN[i*2+1], 'g:', label="ANN")
    plt.plot(AR[i*2], AR[i*2+1], 'b-', label="AR")
    plt.legend(loc=4)
    plt.savefig("CDF_of_MSE_"+str(i+1)+".png", dpi=300, format='png')
