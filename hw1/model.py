import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle


class Model(object):
    def __init__(self, input_data, output_data, name, type, batch_size=64):
        self.name = type + "/" + name
        self.batch_size = batch_size
        self.input_data = input_data
        self.output_data = np.reshape(output_data, (output_data.shape[0], output_data.shape[2]))

        self.input_shape = [self.batch_size, self.input_data.shape[-1]]
        self.output_shape = [self.batch_size, self.output_data.shape[-1]]

        self.input_placeholder = tf.placeholder(tf.float32, shape=self.input_shape)
        self.output_placeholder_true = tf.placeholder(tf.float32, shape=self.output_shape)
        self.output_placeholder_false = self.build_model(self.input_placeholder)

        self.loss = tf.reduce_mean(tf.nn.l2_loss(self.output_placeholder_true - self.output_placeholder_false))
        self.loss_summary = tf.summary.scalar("loss", self.loss)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        self.sess = tf.Session()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def build_model(self, input_placeholder):
        x = tf.layers.dense(input_placeholder, 64, activation=tf.nn.tanh)
        x = tf.layers.dense(x, 32, activation=tf.nn.tanh)
        x = tf.layers.dense(x, self.output_shape[-1])
        return x

    def train(self, epochs=20, train_data=None, test_data=None, number=None):
        if train_data is None and test_data is None:
            train_data, test_data = shuffle(self.input_data, self.output_data, random_state=0)
        else:
            test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[2]))

        dx = tf.data.Dataset.from_tensor_slices(train_data)
        dy = tf.data.Dataset.from_tensor_slices(test_data)

        dcomb = tf.data.Dataset.zip((dx, dy)).repeat(epochs*50).batch(self.batch_size)

        iterator = dcomb.make_one_shot_iterator()
        next_element = iterator.get_next()
        # extract an element
        saver = tf.train.Saver()

        batch_idxs = len(train_data) // self.batch_size

        if number is None:
            writer = tf.summary.FileWriter(self.name)
        else:
            writer = tf.summary.FileWriter(self.name + str(number))

        writer.add_graph(self.sess.graph)

        for epoch in range(epochs):
            for idx in range(batch_idxs):
                el = self.sess.run(next_element)
                batch_train = el[0]
                batch_value = el[1]
                feed_train = {self.input_placeholder: batch_train,
                              self.output_placeholder_true: batch_value}
                self.sess.run(self.optimizer, feed_dict=feed_train)

                if idx % 20 == 0:
                    loss_summary, loss_value = self.sess.run([self.loss_summary, self.loss], feed_dict=feed_train)
                    writer.add_summary(loss_summary, epoch * batch_idxs + idx)
                    print('epoch: %d iter %d: loss: %f' % (epoch, idx, loss_value))
        saver.save(self.sess, self.name + "/behavior_cloning_model")

    def sample(self, input_):
        output = self.sess.run(self.output_placeholder_false,
                               feed_dict={self.input_placeholder: np.repeat(input_[None, :], 64, axis=0)})
        return output[0]
