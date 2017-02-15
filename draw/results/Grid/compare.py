import pickle
import numpy as np
import matplotlib.pyplot as plt

def test_dataset(load, win_i, win_o, ahead_step):
    load_len = len(load)
    load_test_len = int(0.2 * load_len)
    load_train_len = int(0.8 * load_len)
    load = np.asarray(load)
    load_mean = np.mean(load[:load_train_len])
    load -= load_mean
    load_std = np.std(load[:load_train_len])
    load /= load_std
    model_train_len = 26 * 288 / ahead_step / 8
    model_test_len = 2 * 288 / ahead_step + (288 - win_o - win_i) / ahead_step + 1
    tr_num = int(load_train_len / (128 * ahead_step * model_train_len)) * 128
    te_num = int(load_test_len / (64 * ahead_step * model_test_len)) * 64
    print("Unix system train", tr_num, ", test", te_num)
    train_start = load_train_len - tr_num * ahead_step * model_train_len
    test_start = -load_test_len
    X_train = np.asarray([[load[train_start+i*model_train_len*ahead_step+j*ahead_step:
                                train_start+i*model_train_len*ahead_step+j*ahead_step+win_i] 
                                for j in range(model_train_len)] for i in range(tr_num)])
    y_train = np.asarray([[load[train_start+i*model_train_len*ahead_step+j*ahead_step+win_i:
                                train_start+i*model_train_len*ahead_step+j*ahead_step+win_i+win_o] 
                                for j in range(model_train_len)] for i in range(tr_num)])
    X_test = np.asarray([[load[test_start+i*model_test_len*ahead_step+j*ahead_step:
                               test_start+i*model_test_len*ahead_step+j*ahead_step+win_i] 
                               for j in range(model_test_len)] for i in range(te_num)])
    y_test = np.asarray([[load[test_start+i*model_test_len*ahead_step+j*ahead_step+win_i:
                               test_start+i*model_test_len*ahead_step+j*ahead_step+win_i+win_o] 
                               for j in range(model_test_len)] for i in range(te_num)])
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    return X_train, y_train, X_test, y_test, load_std, load_mean
    
def plot_single(X_test, y_test, y_predict):
    plt.figure()
    X = []
    for i in X_test:
        X.append(i)
    X.append(y_test[0])
    plt.plot(X, label="history")
    plt.plot(range(24,30), y_test, label="actual")
    #y_predict[machine][index][5] -= 0.02
    plt.plot(range(24,30), y_predict, label="predict")
    plt.axvline(x=24, ymin=0.2, ymax=0.7, color='y', linestyle='--')
    plt.xlabel("Step")
    plt.ylabel("Hostload")
    plt.ylim([.14, .35])
    plt.legend(loc=1)
    plt.savefig("actual load prediction grid.png", dpi=300, format='png')
    plt.show()

    
with open("./data/axp7.pkl", 'rb') as f:
    grid = pickle.load(f)
input_dim = 24
output_dim = 6
X_train, y_train, X_test, y_test, std_grid, mean_grid = test_dataset(grid, input_dim, 
                                                                     output_dim, input_dim)
    
save_path = "./logits/lstm_"+str(output_dim)+".pkl"    
with open(save_path, 'rb') as input:
    a = pickle.load(input)
    y_predict = a.reshape(y_test.shape)

X_test = (X_test * std_grid) + mean_grid
y_test = (y_test * std_grid) + mean_grid
y_predict = (y_predict * std_grid) + mean_grid

print np.mean((y_predict-y_test)**2)

index_0 = 0
index_1 = 23
plot_single(X_test[index_0][index_1], 
            y_test[index_0][index_1], 
            y_predict[index_0][index_1])

