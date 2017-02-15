import pickle
from utils import read_data
import numpy as np
import matplotlib.pyplot as plt
    
data_path = "/home/tyrion/lannister/1024/tyrion.pkl"
input_dim = 24
#output_dim_arr = [6,12,18,24,30,36]
output_dim_arr = [6]
for i in output_dim_arr:
    output_dim = i
    print(output_dim)
    X_train, y_train, X_test, y_test, cpu_load_mean, cpu_load_std = read_data(data_path, input_dim, output_dim, input_dim)
    
    save_path = "./logits/rnn_"+str(output_dim)+".pkl"    
    with open(save_path, 'rb') as input:
        a = pickle.load(input)
        y_predict = a.reshape(y_test.shape)
    
    X_test = (X_test * cpu_load_std) + cpu_load_mean
    y_test = (y_test * cpu_load_std) + cpu_load_mean
    y_predict = (y_predict * cpu_load_std) + cpu_load_mean
    
    dict1 = {}   
    for j in range(1024):
        mse = np.mean((y_predict[j] - y_test[j])**2)
        dict1[j] = mse
    dict2 = sorted(dict1.items(), lambda x, y: cmp(x[1], y[1]))
#    id = []
#    for i in xrange(30):
#        id.append(dict2[i][0])
#    for i in xrange(30):
#        plot_single(X_test, y_test, y_predict, id[i])

def plot_single(X_test, y_test, y_predict, machine, index=None, ran=None):
    X = X_test[machine]
    a = []
    for i in X:
        a.extend(i)
    a = np.asarray(a)
    plt.figure()
    if ran:
        plt.plot(a[ran], label="history")
        plt.plot(range(24,30), y_test[machine][index], label="actual")
        y_predict[machine][index][5] += 0.02
        plt.plot(range(24,30), y_predict[machine][index], label="predict")
        plt.axvline(x=24, ymin=0.2, ymax=0.7, color='y', linestyle='--')
        plt.xlabel("Step")
        plt.ylabel("Hostload")
        plt.ylim([.0, .25])
        plt.legend(loc=1)
        plt.savefig("actual load prediction cloud.png", dpi=300, format='png')
        plt.show()
    else:
        plt.plot(a)
        plt.title(str(machine))
        
machine = 708
index = 0
plot_single(X_test, y_test, y_predict, machine, index, range(24*index, 24*index+30))



