import numpy as np
    
def test_dataset(load, win_i, win_o, ahead_step):
    load_len = len(load)
    load_test_len = int(0.2 * load_len)
    load_train_len = int(0.8 * load_len)
    load = np.asarray(load)
    load -= np.mean(load[:load_train_len])
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
    return X_train, y_train, X_test, y_test, load_std
