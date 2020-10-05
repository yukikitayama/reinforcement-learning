import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense


class Embedding(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(Embedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.layer = Dense(units=self.embedding_dim, activation='relu')

    def call(self, batch_size, num_quantile, tau_min, tau_max):
        # Get an array of uniform random variables length batch_size * num_quantile
        sample = tf.random.uniform(shape=[batch_size * num_quantile, 1],
                                   minval=tau_min, maxval=tau_max, dtype=tf.float32)
        # Convert shape from (batch_size * num_quantile, 1)
        # to (batch_size * num_quantile, embedding_dim)
        sample_tile = tf.tile(input=sample, multiples=[1, self.embedding_dim])
        # embedding.shape is the same as sample_tile.shape
        embedding = tf.cos(
            tf.cast(x=tf.range(start=0, limit=self.embedding_dim, delta=1),
                    dtype=tf.float32) * np.pi * sample_tile
        )
        # ?
        embedding_out = self.layer(embedding)
        return embedding_out, sample


class IQN(tf.keras.Model):
    def __init__(self, num_actions, embedding_dim):
        super(IQN, self).__init__()
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.embedding_out = Embedding(self.embedding_dim)
        self.hidden1 = Dense(units=64, activation='relu')
        self.hidden2 = Dense(units=64, activation='relu')
        self.hidden3 = Dense(units=64, activation='relu')
