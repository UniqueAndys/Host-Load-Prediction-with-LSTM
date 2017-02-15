import pickle
import numpy as np
    
def zero_center(cpu_load):
    cpu_load = np.asarray(cpu_load)
    cpu_load_mean = np.mean(cpu_load[:,:24*12*26])
    cpu_load_std = np.std(cpu_load[:,:24*12*26])
    cpu_load -= cpu_load_mean
    cpu_load /= cpu_load_std
    return (cpu_load, cpu_load_mean, cpu_load_std)

def contextwin(cpu_load, win_i, win_o, ahead_step):
    m, cpu_load_mean, cpu_load_std = zero_center(cpu_load)
    a = 26
    b = 3
    train_len = a * 288 / ahead_step
    test_len = (b-1) * 288 / ahead_step + (288 - win_o - win_i) / ahead_step + 1
    train_start = win_i
    test_start = a*288 + win_i
    
    train_x = np.asarray([[m[i][train_start+j*ahead_step-win_i:train_start+j*ahead_step] 
                                for j in range(train_len)] for i in range(len(m))],dtype=np.float32)
    train_y = np.asarray([[m[i][train_start+j*ahead_step:train_start+j*ahead_step+win_o]
                                for j in range(train_len)] for i in range(len(m))],dtype=np.float32)
    test_x = np.asarray([[m[i][test_start+j*ahead_step-win_i:test_start+j*ahead_step] 
                                for j in range(test_len)] for i in range(len(m))],dtype=np.float32)
    test_y = np.asarray([[m[i][test_start+j*ahead_step:test_start+j*ahead_step+win_o] 
                                for j in range(test_len)] for i in range(len(m))],dtype=np.float32)

    return (train_x, train_y, test_x, test_y, cpu_load_mean, cpu_load_std)

def read_data(_data_path, win_i, win_o, ahead_step):
    data_path = _data_path
    print("Reading pkl data...")
    input_machine = open(data_path,'rb')
    cpu_load = pickle.load(input_machine)
    input_machine.close()
    print("Loading data...")
    X_train, y_train, X_test, y_test, cpu_load_mean, cpu_load_std = contextwin(cpu_load, win_i, win_o, ahead_step)
    
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    print(cpu_load_mean)
    print(cpu_load_std)
    
    return (X_train, y_train, X_test, y_test, cpu_load_mean, cpu_load_std)

def truncated_index(long_period, num, reverse):
    index = []
    single_period_len = int(long_period/num)
    for i in range(num):
        if reverse:
            j = num-1 - i
        else:
            j = i
        index.append(range(j*single_period_len, (j+1)*single_period_len))
    return single_period_len, index
    
def test_dataset(load, win_i, win_o, ahead_step):
    load_len = len(load)
    load_test_len = int(0.2 * load_len)
    load_train_len = int(0.8 * load_len)
    load = np.asarray(load)
    load -= np.mean(load[:load_train_len])
    load_std = np.std(load[:load_train_len])
    load /= load_std
    model_train_len = 26 * 288 / ahead_step
    model_test_len = 2 * 288 / ahead_step + (288 - win_o - win_i) / ahead_step + 1
    tr_num = int(load_train_len / (16 * ahead_step * model_train_len)) * 16
    te_num = int(load_test_len / (16 * ahead_step * model_test_len)) * 16
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
    return X_train, y_train, X_test, y_test, load_std
