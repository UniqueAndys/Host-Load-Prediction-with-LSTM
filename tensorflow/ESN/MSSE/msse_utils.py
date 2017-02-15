import pickle
import numpy as np
    
def zero_center(cpu_load):
    cpu_load = np.asarray(cpu_load)
    cpu_load_mean = np.mean(cpu_load[:,:24*12*26])
    cpu_load_std = np.std(cpu_load[:,:24*12*26])
    cpu_load -= cpu_load_mean
    cpu_load /= cpu_load_std
    return (cpu_load, cpu_load_mean, cpu_load_std)
    
def calcu_mean(actual_data, start, base_seg, n):
    seg_mean = []
    for i in range(n):
        seg_mean.append(np.mean(actual_data[start:start+base_seg*2**i]))
    return seg_mean

def contextwin(cpu_load, win_i, n, ahead_step):
    m, cpu_load_mean, cpu_load_std = zero_center(cpu_load)
    a = 26
    b = 3
    train_len = a * 288 / ahead_step
    test_len = (b-1) * 288 / ahead_step + (288 - 2**(n-1)) / ahead_step
    train_start = win_i
    test_start = a*288 + win_i
    
    train_x = np.asarray([[m[i][train_start+j*ahead_step-win_i:train_start+j*ahead_step] 
                        for j in range(train_len)] for i in range(len(m))],dtype=np.float32)
    train_y = np.asarray([[calcu_mean(m[i], train_start+j*ahead_step, 1, n)
                        for j in range(train_len)] for i in range(len(m))],dtype=np.float32)
    test_x = np.asarray([[m[i][test_start+j*ahead_step-win_i:test_start+j*ahead_step] 
                        for j in range(test_len)] for i in range(len(m))],dtype=np.float32)
    test_y = np.asarray([[calcu_mean(m[i], test_start+j*ahead_step, 1, n)
                        for j in range(test_len)] for i in range(len(m))],dtype=np.float32)

    return (train_x, train_y, test_x, test_y, cpu_load_mean, cpu_load_std)

def read_data(_data_path, win_i, n, ahead_step):
    data_path = _data_path
    print("Reading pkl data...")
    input_machine = open(data_path,'rb')
    cpu_load = pickle.load(input_machine)
    input_machine.close()
    print("Loading data...")
    X_train, y_train, X_test, y_test, cpu_load_mean, cpu_load_std = contextwin(cpu_load, win_i, n, ahead_step)
    
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    print(cpu_load_mean)
    print(cpu_load_std)
    
    return (X_train, y_train, X_test, y_test, cpu_load_mean, cpu_load_std)

#data_path = "/home/tyrion/Documents/Cloud Computing/python/data/72/tyrion.pkl"
#X_train, y_train, X_test, y_test, _, cpu_load_std = read_data(data_path, 24, 6, 24) 