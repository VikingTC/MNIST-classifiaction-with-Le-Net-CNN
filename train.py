import tensorflow as tf
import numpy as np
from functools import partial
from datetime import datetime
import time
from sklearn.metrics import  accuracy_score

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

print(X_train.shape)
def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

# build model

n_f5 = 84
n_ouputs = 10
height = 28
width = 28
channels = 1
n_inputs = height * width

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name="y")

with tf.name_scope('LeNet-cnn'):
    c1 = tf.layers.conv2d(X_reshaped, filters=6, kernel_size=5, strides=[1, 1], padding='SAME', activation=tf.nn.relu, name='conv1')
    s2 = tf.nn.avg_pool(c1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
    c3 = tf.layers.conv2d(s2, filters=16, kernel_size=5, strides=[1, 1], padding='SAME', activation=tf.nn.relu, name='conv3')
    s4 = tf.nn.avg_pool(c3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool4')
    c5 = tf.layers.conv2d(s4, filters=32, kernel_size=5, strides=[1, 1], padding='SAME', activation=tf.nn.relu, name='conv5')
    c5_reshape = tf.reshape(c5, shape=[-1, 32*7*7])
    f5 = tf.layers.dense(c5_reshape, n_f5, activation=tf.nn.relu, name='fc5')
    logits = tf.layers.dense(f5, n_ouputs, name='logits')

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')
    loss_sum = tf.summary.scalar('loss', loss)

learning_rate = 0.01
with tf.name_scope('train'):
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
    accuracy_sum = tf.summary.scalar('accuracy', accuracy)

    y_proba = tf.nn.softmax(logits)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'batch_norm_train_tf_logdir'
logdir = '{}/run-{}'.format(root_logdir, now)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

n_epochs = 40
batch_size = 50
best_loss = np.infty
next_epoch = 0
max_epoch = 5

t0 = time.time()

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val = loss.eval(feed_dict={X: X_valid, y: y_valid})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        accuracy_sum_val = accuracy_sum.eval(feed_dict={X: X_valid, y: y_valid})
        loss_sum_val = loss_sum.eval(feed_dict={X: X_valid, y: y_valid})
        file_writer.add_summary(accuracy_sum_val, epoch)
        file_writer.add_summary(loss_sum_val, epoch)
        print(epoch, '\tValidation loss:', loss_val, "\tValidation accuracy:", accuracy_val)
        if loss_val < best_loss:
            best_loss = loss_val
            next_epoch = 0
            saver_path = saver.save(sess, 'current_model.ckpt')
        else:
            next_epoch += 1
            if next_epoch > max_epoch:
                print('Early stopping!')
                break
    save_path = saver.save(sess, "./my_model_final.ckpt")

    accuracy_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
file_writer.close()
t1 = time.time()
print('Took time :', t1-t0)
print('Accuracy on test set:', accuracy_test)