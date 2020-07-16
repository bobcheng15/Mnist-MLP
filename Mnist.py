import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import progressbar
import datetime
def one_hot(label):
    oh_label = np.zeros((label.shape[0], 10))  
    index = 0
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
    for l in label:
        oh_label[index][int(l)] = 1
        index = index + 1
        bar.update(index)
    return oh_label


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.FileWriter(train_log_dir)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train, X_test = X_train / 255.0, X_test / 255.0
y_train = one_hot(y_train)
y_test = one_hot(y_test)


print("Training dataset: {}".format(X_train.shape))
print("Testing dataset: {}".format(X_test.shape))
print("label: {}".format(y_train.shape))

class Dataset:
    def __init__(self, batch_size=32):
        with tf.variable_scope('dataset'):
            self.dataset_x = tf.placeholder(tf.float32, [None, 28, 28])
            self.dataset_y = tf.placeholder(tf.float32, [None, 10])
            dataset = tf.data.Dataset.from_tensor_slices((self.dataset_x, self.dataset_y))
            dataset = dataset.shuffle(10000)
            dataset = dataset.batch(batch_size)
            self.iterator = dataset.make_initializable_iterator()
            self.batch = self.iterator.get_next()
        
    def init_dataset(self, sess, x, y):
        sess.run(self.iterator.initializer, feed_dict={
            self.dataset_x: x,
            self.dataset_y: y
        })
        
    def next_batch(self, sess):
        return sess.run(self.batch)
class MLP_with_dataset:
    def __init__(self, learning     
class MLP_with_placeholder:
    def __init__(self, learning_rate=0.01):
       with tf.variable_scope('mlp_with_placeholder'):
            self.x = tf.placeholder(tf.float32, [None, 28, 28])
            self.y_true = tf.placeholder(tf.float32, [None, 10])

            self._build_model(learning_rate)
    
    def _build_model(self, lr):
            with tf.variable_scope('model'):
                self.flat = tf.layers.Flatten()(self.x)
                self.layer1 = tf.layers.Dense(500, activation='relu')(self.flat)
                self.y_pred = tf.layers.Dense(10, activation='softmax')(self.layer1)            
            with tf.variable_scope('loss'):
                self.loss = tf.keras.losses.MSE(self.y_true, self.y_pred)
            with tf.variable_scope('training_step'):
                self.train_step = tf.train.GradientDescentOptimizer(lr).minimize(self.loss)

            with tf.variable_scope('test'):
                self.correct_prediction = tf.equal(tf.argmax(self.y_true, 1), tf.argmax(self.y_pred, 1))
                self.test_step = tf.reduce_sum(tf.cast(self.correct_prediction, tf.float32))
            
    def train(self, sess, x, y):
        sess.run(self.train_step, feed_dict={
            self.x: x,
            self.y_true: y
        })
        
    def test(self, sess, x, y):
        return sess.run(self.test_step, feed_dict={
            self.x: x,
            self.y_true: y
        })
    
with tf.name_scope('performance'):
    tf_accuracy_ph = tf.placeholder(tf.float32,shape=None, name='accuracy_summary')
    tf_accuracy_summary = tf.summary.scalar('accuracy', tf_accuracy_ph)


dataset = Dataset()
mlp_with_placeholder = MLP_with_placeholder()

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    start = time.time()
    sess.run(tf.global_variables_initializer())
    
    for e in range(10):
        dataset.init_dataset(sess, X_train, y_train)
        while True:
            try:
                x_batches, y_batches = dataset.next_batch(sess)
            except:
                break
            mlp_with_placeholder.train(sess, x_batches, y_batches)
            
        dataset.init_dataset(sess, X_test, y_test)
        acc_num = 0
        while True:
            try:
                x_batches, y_batches = dataset.next_batch(sess)
            except:
                break
            acc_num += mlp_with_placeholder.test(sess, x_batches, y_batches)
        acc_percent = acc_num / len(X_test) * 100
        s = sess.run(tf_accuracy_summary , feed_dict={tf_accuracy_ph: acc_percent})
        train_summary_writer.add_summary(s, e)

        print('Epoch {}, Accuracy: {:.2f}%'.format(e, acc_num / len(X_test) * 100), end='\r')      
    print('\ntime per epoch: {:.3f} sec'.format((time.time() - start)/10))
