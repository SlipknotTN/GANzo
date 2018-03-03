import tensorflow as tf


def generator(z, units=128, out_dim=28*28, alpha=0.01):

    # Hidden layer
    h1 = tf.layers.dense(inputs=z, units=units, activation=None, name="dense_1")
    # Leaky ReLU
    h1 = tf.maximum(tf.scalar_mul(alpha, h1), h1)

    # tanh output
    dense = tf.layers.dense(inputs=h1, units=out_dim, activation=None, name="dense_2")
    out = tf.tanh(dense)

    return out


def discriminator(x, unusedConditioning, units=128, alpha=0.01):

    # Hidden layer
    h1 = tf.layers.dense(inputs=x, units=units, activation=None, name="dense_1")
    # Leaky ReLU
    h1 = tf.maximum(tf.scalar_mul(alpha, h1), h1)

    # Logits and sigmoid output
    logits = tf.layers.dense(inputs=h1, units=1, activation=None, name="dense_2")
    out = tf.sigmoid(logits)

    return out
