import tensorflow as tf
import numpy as np
import os

class Recognizer:
    def __init__(self, feature_count, class_count, max_length):
        self.step = 0
        self.learning_rate = 0.2
        batch_size = None
        num_mem_cells = 64
        conv_width = 4
        conv_height = 1
        conv_in_channels = feature_count
        conv_out_channels = 128
        # MODEL VARIABLES   
        with tf.variable_scope('variables'):     
            # projection matrix
            self.weight = tf.Variable(tf.truncated_normal([num_mem_cells, class_count], stddev = 0.1))
            self.bias = tf.Variable(tf.constant(0.1, shape=[class_count]))
            self.conv_filter = tf.Variable(tf.truncated_normal([conv_height, conv_width, conv_in_channels, conv_out_channels], stddev=0.1))
            self.conv_biases = tf.Variable(tf.constant(0.1, shape=[conv_out_channels]))
            
        # recurent cells
        cell = tf.contrib.rnn.LSTMCell(num_mem_cells, state_is_tuple=True)
        cell = tf.contrib.rnn.MultiRNNCell([cell] * 3, state_is_tuple=True)
        
        # TRAINING GRAPH
        with tf.variable_scope('training'):
        # INPUT DATA
            with tf.variable_scope('input'):
                self.examples = tf.placeholder(tf.float32, [batch_size, max_length, feature_count], name="examples")
                self.labels = tf.placeholder(tf.int32, [batch_size, max_length], name="labels")
                self.lengths = tf.placeholder(tf.int32, [batch_size], name="lengths")
                self.keep_prob = tf.placeholder(tf.float32, (), name='keep_prob')
                
            with tf.variable_scope('operations'):
                conv_input = tf.reshape(self.examples, (-1, 1, max_length, conv_in_channels))
                conv = tf.nn.conv2d(conv_input, self.conv_filter, [1, 1, 1, 1], 'SAME', True)
                conv = tf.nn.bias_add(conv, self.conv_biases)
                conv = tf.nn.sigmoid(conv)
                conv = tf.nn.dropout(conv, self.keep_prob)

                cell_in = tf.reshape(conv, (-1, max_length, conv_out_channels))
                # TRAINING AND VALIDATION OPERATIONS
                cell_out, cell_state = tf.nn.dynamic_rnn(
                    cell, cell_in, dtype=tf.float32, sequence_length=self.lengths)
            
                flat = tf.reshape(cell_out, (-1, num_mem_cells))
                prediction = tf.matmul(flat, self.weight) + self.bias
                prediction = tf.reshape(prediction, (tf.shape(self.examples)[0], max_length, class_count), name="predictions")
                
                self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=self.labels, name="softmax_cross_entropy")
                batch_size = tf.shape(self.cross_entropy)[0]
                index = tf.range(0, batch_size) * max_length + (self.lengths - 1)
                flat = tf.reshape(self.cross_entropy, [-1])
                self.cross_entropy = tf.gather(flat, index)
                self.cross_entropy = tf.reduce_mean(self.cross_entropy, name='cross_entropy')
                #mask = tf.sequence_mask(self.lengths,  max_length)
                #self.cross_entropy = tf.reduce_mean(tf.boolean_mask(self.cross_entropy, mask), name="cross_entropy")
                
                optimizer = tf.train.AdamOptimizer()
                #self.learning_rate_var = tf.placeholder(tf.float32, shape=(), name='learning_rate')
                #optimizer = tf.train.GradientDescentOptimizer(self.learning_rate_var)
                self.optimize = optimizer.minimize(self.cross_entropy, name="optimize")
        
                # EXAMINATION OPERATION
                indices = tf.expand_dims(self.lengths - 1, 1)
                sequence = tf.expand_dims(tf.range(tf.shape(self.lengths)[0], dtype=tf.int32), 1)
                indices = tf.concat([sequence, indices], 1, name="Last_from_each_row_indices")

                self.total_predictions = tf.nn.softmax(prediction) 
                self.prediction = tf.nn.softmax(tf.gather_nd(prediction, indices))
                self.correct_percentage = tf.to_int32(tf.argmax(self.prediction, axis=1))
                self.correct_percentage = tf.equal(self.correct_percentage, tf.gather_nd(self.labels, indices))
                self.correct_percentage = tf.reduce_mean(tf.to_float(self.correct_percentage))

        self.sess = tf.Session()
        self.reset()
        #writer = tf.summary.FileWriter("./log", self.sess.graph)
        #writer.close()

    def reset(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)


    def train(self, examples, labels, lengths):
        feed = {
            self.examples : examples,
            self.labels : labels,
            self.lengths : lengths,
            self.keep_prob : 0.9,
        }
        _, ret = self.sess.run([self.optimize, self.cross_entropy], feed)
        return ret

    def test(self, examples, labels, lengths):
        feed = {
            self.examples : examples,
            self.labels : labels,
            self.lengths : lengths,
            self.keep_prob : 1.0
        }
        return self.sess.run([self.cross_entropy, self.correct_percentage, self.prediction, self.total_predictions], feed)

    def save(self):
        save_dir = "./save/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        saver = tf.train.Saver()
        saver.save(self.sess, save_dir + "model", global_step=self.step)
        self.step += 1