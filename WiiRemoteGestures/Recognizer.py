import tensorflow as tf
import numpy as np


class Recognizer:
    def __init__(self, feature_count, class_count, max_length):
        num_mem_cells = 20
        # MODEL VARIABLES   
        with tf.variable_scope('variables'):     
            # projection matrix
            self.weight = tf.Variable(tf.truncated_normal([num_mem_cells, 2], stddev = 0.1))
            self.bias = tf.Variable(tf.constant(0.1, shape=[2]))
            
        # recurent cells
        cell = tf.contrib.rnn.LSTMCell(num_mem_cells, state_is_tuple=True)
        cell = tf.contrib.rnn.MultiRNNCell([cell] * 3, state_is_tuple=True)
        
        # TRAINING GRAPH
        with tf.variable_scope('training'):
        # INPUT DATA
            with tf.variable_scope('input'):
                self.examples = tf.placeholder(tf.float32, [None, max_length, feature_count], name="examples")
                self.labels = tf.placeholder(tf.float32, [None, max_length, class_count], name="labels")
                self.lengths = tf.placeholder(tf.int32, [None], name="lengths")
                
            
            with tf.variable_scope('operations'):
                # TRAINING AND VALIDATION OPERATIONS
                cell_out, cell_state = tf.nn.dynamic_rnn(
                    cell, self.examples, dtype=tf.float32, sequence_length=self.lengths)
            
                flat = tf.reshape(cell_out, (-1, num_mem_cells))
                prediction = tf.matmul(flat, self.weight) + self.bias
                prediction = tf.reshape(prediction, (tf.shape(self.examples)[0], max_length, 2))
                
                self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.labels)
                self.cross_entropy = tf.reduce_mean(self.cross_entropy, name="cross_entropy")
                
                optimizer = tf.train.AdamOptimizer()
                self.optimize = optimizer.minimize(self.cross_entropy, name="optimize")
        
                # EXAMINATION OPERATION
                self.prediction = tf.nn.softmax(prediction)

        init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init_op)

    def train(self, examples, labels, lengths):
        feed = {
            self.examples : examples,
            self.labels : labels,
            self.lengths : lengths
        }
        self.sess.run(self.optimize, feed)

    def test(self, examples, labels):
        pass

        