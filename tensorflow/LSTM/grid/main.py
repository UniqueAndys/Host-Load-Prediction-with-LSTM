from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from utils_grid import test_dataset
import rnn_cell

import numpy as np
import time
import pickle

#from pastalog import Log

flags = tf.flags
logging = tf.logging
flags.DEFINE_string("data_path", "/home/tyrion/lannister/1024/tyrion.pkl", 
                    "The path of host load data")
flags.DEFINE_integer("input_dim", 24, "The length of history window")
flags.DEFINE_integer("hidden_dim", 128, "The length of hidden layer size")
flags.DEFINE_integer("output_dim", 6, "The length of prediction window")
flags.DEFINE_integer("batch_size", 32, "Mini-batch size")
flags.DEFINE_integer("test_batch_size", 32, "Mini-batch size of testing data")
flags.DEFINE_integer("epoch", 60, "The total epochs")
flags.DEFINE_float("lr", 0.05, "Learning rate")
flags.DEFINE_string("model", "lstm", "The RNN type")
flags.DEFINE_string("grid", "axp7", "The machine of Grid")
flags.DEFINE_integer("layer", 1, "The layers of model")
flags.DEFINE_float("keep_prob", 1.0, "keep prob")
flags.DEFINE_integer("max_grad_norm", 5, "max grad norm")
FLAGS = flags.FLAGS

class RNNModel(object):
    def __init__(self, is_training, length, batch_size):
        self.batch_size = batch_size
        self.num_steps = num_steps = length
        hidden_size = FLAGS.hidden_dim
        
        self._input_data = tf.placeholder(tf.float32, [batch_size, None, FLAGS.input_dim])
        self._targets = tf.placeholder(tf.float32, [batch_size, None, FLAGS.output_dim])
        
        if FLAGS.model == "rnn":
            vanilla_rnn_cell = rnn_cell.BasicRNNCell(num_units=FLAGS.hidden_dim)
            if is_training and FLAGS.keep_prob < 1:
                vanilla_rnn_cell = rnn_cell.DropoutWrapper(vanilla_rnn_cell, 
                                                           output_keep_prob=FLAGS.keep_prob)
            if FLAGS.layer == 1:
                cell = vanilla_rnn_cell
            elif FLAGS.layer == 2:
                cell = rnn_cell.MultiRNNCell([vanilla_rnn_cell] * 2)
        elif FLAGS.model == "lstm":
            lstm_cell = rnn_cell.BasicLSTMCell(num_units=FLAGS.hidden_dim,
                                               forget_bias=1.0)
            if is_training and FLAGS.keep_prob < 1:
                lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, 
                                                    output_keep_prob=FLAGS.keep_prob)
            if FLAGS.layer == 1:
                cell = lstm_cell
            elif FLAGS.layer == 2:
                cell = rnn_cell.MultiRNNCell([lstm_cell] * 2)
        elif FLAGS.model == "gru":
            gru_cell = rnn_cell.GRUCell(num_units=FLAGS.hidden_dim)
            if is_training and FLAGS.keep_prob < 1:
                gru_cell = rnn_cell.DropoutWrapper(gru_cell, 
                                                   output_keep_prob=FLAGS.keep_prob)
            cell = gru_cell
        else:
            raise ValueError("Invalid model: %s", FLAGS.model)
        
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(self._input_data[:, time_step, :], state)
                outputs.append(cell_output)
        self._final_state = state
        
        hidden_output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])
        
        V_1 = tf.get_variable("v_1", shape=[hidden_size, FLAGS.output_dim],
          initializer=tf.random_uniform_initializer(-tf.sqrt(1./hidden_size),tf.sqrt(1./hidden_size)))
        b_1 = tf.get_variable("b_1", shape=[FLAGS.output_dim], initializer=tf.constant_initializer(0.1))
        logits = tf.add(tf.matmul(hidden_output, V_1), b_1)
        
        target = tf.reshape(self._targets, [-1, FLAGS.output_dim])
        training_loss = tf.reduce_sum(tf.pow(logits-target, 2)) / 2        
        mse = tf.reduce_mean(tf.pow(logits-target, 2))        
        self._cost = mse
        
        if not is_training:
            return
        
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(training_loss, tvars), FLAGS.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))
        
    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))
        
    @property
    def input_data(self):
        return self._input_data
        
    @property
    def targets(self):
        return self._targets
        
    @property
    def initial_state(self):
        return self._initial_state
        
    @property
    def cost(self):
        return self._cost
        
    @property
    def final_state(self):
        return self._final_state
        
    @property
    def lr(self):
        return self._lr
        
    @property
    def train_op(self):
        return self._train_op
    
def run_train_epoch(session, m, data_x, data_y, eval_op):
    costs = []
    states = []
    for i in xrange(int(len(data_y) / FLAGS.batch_size)):
        cost, state, _ = session.run(
            [m.cost, m.final_state, eval_op],
            {m.input_data: data_x[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size],
             m.targets: data_y[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]})
        costs.append(cost)
        states.append(state)
    return (np.mean(costs), states)
    
def run_test_epoch(session, m, data_x, data_y, eval_op):
    costs = []
    states = []
    for i in xrange(int(len(data_y) / FLAGS.test_batch_size)):
        cost, state, _ = session.run(
            [m.cost, m.final_state, eval_op],
            {m.input_data: data_x[i*FLAGS.test_batch_size:(i+1)*FLAGS.test_batch_size],
             m.targets: data_y[i*FLAGS.test_batch_size:(i+1)*FLAGS.test_batch_size]})
#             m.targets: data_y[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]})
        costs.append(cost)
        states.append(state)
    return (np.mean(costs), states)    

def main(_):
    print("===============================================================================")
    print("The input_dim is", FLAGS.input_dim, "The hidden_dim is", FLAGS.hidden_dim)
    print("The output_dim is", FLAGS.output_dim)
    print("The keep_prob is", FLAGS.keep_prob, "The batch_size is", FLAGS.batch_size)
    print("The model is", FLAGS.model, "The machine is", FLAGS.grid)
#    with open("./data/axp0.pkl", 'rb') as f:
#        axp0 = pickle.load(f)
#    X_train, y_train, X_test, y_test, std_grid = test_dataset(axp0, 
#                                        FLAGS.input_dim, FLAGS.output_dim, FLAGS.input_dim)

#    with open("./data/axp7.pkl", 'rb') as f:
#        axp7 = pickle.load(f)
#    X_train, y_train, X_test, y_test, std_grid = test_dataset(axp7, 
#                                        FLAGS.input_dim, FLAGS.output_dim, FLAGS.input_dim)

#    with open("./data/sahara.pkl", 'rb') as f:
#        sahara = pickle.load(f)
#    X_train, y_train, X_test, y_test, std_grid = test_dataset(sahara, 
#                                        FLAGS.input_dim, FLAGS.output_dim, FLAGS.input_dim)

#    with open("./data/themis.pkl", 'rb') as f:
#        themis = pickle.load(f)
#    X_train, y_train, X_test, y_test, std_grid = test_dataset(themis, 
#                                        FLAGS.input_dim, FLAGS.output_dim, FLAGS.input_dim)

    with open("./data/"+FLAGS.grid+".pkl", 'rb') as f:
        grid = pickle.load(f)
    X_train, y_train, X_test, y_test, std_grid = test_dataset(grid, 
                                        FLAGS.input_dim, FLAGS.output_dim, FLAGS.input_dim)
    
    train_len = X_train.shape[1]
    print("train length", train_len)
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope("model", reuse=None):
            m_train = RNNModel(is_training=True, length=X_train.shape[1], 
                               batch_size=FLAGS.batch_size)
        with tf.variable_scope("model", reuse=True):
            m_test = RNNModel(is_training=False, length=X_test.shape[1], 
                              batch_size=FLAGS.test_batch_size)
            
        tf.initialize_all_variables().run()
        
        #log_a = Log('http://localhost:8120','modelA')
        # pastalog --serve 8120
        
        scale = std_grid ** 2
        test_best = 0.0
        training_time = []
        for i in range(FLAGS.epoch):
            if i < FLAGS.epoch/3:
                lr_decay = 1
            elif i < FLAGS.epoch*2/3:
                lr_decay = 0.1
            else:
                lr_decay = 0.01
            m_train.assign_lr(session, FLAGS.lr * lr_decay)
            start = time.time()
            train_loss, _ = run_train_epoch(session, m_train, X_train, 
                                                      y_train, m_train.train_op)
            finish = time.time()
            training_time.append(finish - start)
            test_loss, _ = run_test_epoch(session, m_test, X_test, y_test, 
                                          tf.no_op())
            if i == 0:
                test_best = test_loss
            if test_loss < test_best:
                test_best = test_loss
#            print("epoch:%3d, lr %.5f, train_loss_1 %.6f, train_loss_2 %.6f, test_loss %.6f" %
#                    (i + 1, session.run(m_train.lr), train_loss_1*scale, 
#                     train_loss_2*scale, test_loss*scale))
            #print(np.asarray(train_loss_list)*scale)
            print("epoch:%3d, lr %.5f, train_loss %.6f, test_loss %.6f, speed %.2f seconds/epoch" 
                  % (i + 1, session.run(m_train.lr), train_loss*scale, 
                     test_loss*scale, training_time[i]))
            #log_a.post("trainLoss", value=float(train_loss), step=i)
            #log_a.post("testLoss", value=float(test_loss), step=i)
            if i == FLAGS.epoch - 1:
                print("Best test loss %.6f" % (test_best*scale))
                print("Average %.4f seconds for one epoch" % (np.mean(training_time)))
            
    print("The input_dim is", FLAGS.input_dim, "The hidden_dim is", FLAGS.hidden_dim)
    print("The output_dim is", FLAGS.output_dim)
    print("The keep_prob is", FLAGS.keep_prob, "The batch_size is", FLAGS.batch_size)
    print("The model is", FLAGS.model, "The machine is", FLAGS.grid)
    print("===============================================================================")    

if __name__ == "__main__":
    tf.app.run()
