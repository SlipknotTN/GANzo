import tensorflow as tf


def generator(z, out_channel_dim=3, alpha=0.2, keepProb=0.7, isTraining=True):
    """
    Create the generator network
    :param z: Input z
    :param out_channel_dim: The number of channels in the output image
    :param is_train: Boolean if generator is being used for training
    :return: The tensor output of the generator
    """

    # 1st block -> 4x4x512
    # First fully connected layer
    # From input (1D to 1D of 4*4*512)
    x = tf.layers.dense(z, units=(4 * 4 * 512))
    # Reshahe the fully connected result (HWC)
    x1 = tf.reshape(x, shape=(-1, 4, 4, 512))
    # Batch Norm + Leaky ReLU
    x1 = tf.layers.batch_normalization(x1, training=isTraining)
    x1 = tf.maximum(alpha * x1, x1)
    x1 = tf.nn.dropout(x1, keep_prob=keepProb)

    # 2nd block -> 8x8x256
    # Strides 2 with padding 'same' give the output 8x8 (for the filters we explicitly set 256)
    x2 = tf.layers.conv2d_transpose(x1, filters=256, kernel_size=(3, 3), strides=2,
                                    padding='same', activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
    # Batch Norm + Leaky ReLU
    x2 = tf.layers.batch_normalization(x2, training=isTraining)
    x2 = tf.maximum(alpha * x2, x2)
    x2 = tf.nn.dropout(x2, keep_prob=keepProb)

    # 3rd block -> 16x16x128
    # Strides 2 with padding 'same' give the output 16x16 (for the filters we explicitly set 128)
    x3 = tf.layers.conv2d_transpose(x2, filters=128, kernel_size=(3, 3), strides=2,
                                    padding='same', activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
    x3 = tf.layers.batch_normalization(x3, training=isTraining)
    x3 = tf.maximum(alpha * x3, x3)
    x3 = tf.nn.dropout(x3, keep_prob=keepProb)

    # 4th block -> 32x32xCH (no batch normalization and use tanh activation)
    logits = tf.layers.conv2d_transpose(x3, filters=out_channel_dim, kernel_size=(3, 3), strides=2,
                                        padding='same', activation=None,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())

    # Output layer, 32x32xCH
    out = tf.tanh(logits)

    # Probably need to split images
    tf.summary.image(
         "generated",
         out,
         max_outputs=1,
         collections=None,
         family=None
    )

    return out


def discriminator(x, unusedConditioning, alpha=0.2, keepProb=0.7, isTraining=True):

    # Input layer is 32x32xCH (3 or 1) -> 16x16x64
    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(3, 3), strides=2, activation=None,
                         padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
    # No batch normalization, better results
    x = tf.maximum(alpha * x, x)
    x = tf.nn.dropout(x, keep_prob=keepProb)

    # -> 8x8x128
    x1 = tf.layers.conv2d(x, filters=128, kernel_size=(3, 3), strides=2, activation=None,
                          padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
    x1 = tf.layers.batch_normalization(x1, training=isTraining)
    x1 = tf.maximum(alpha * x1, x1)
    x1 = tf.nn.dropout(x1, keep_prob=keepProb)

    # -> 4x4x256
    x2 = tf.layers.conv2d(x1, filters=256, kernel_size=(3, 3), strides=2, activation=None,
                          padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
    x2 = tf.layers.batch_normalization(x2, training=isTraining)
    x2 = tf.maximum(alpha * x2, x2)
    x2 = tf.nn.dropout(x2, keep_prob=keepProb)

    # Reshape for final dense layer -> units = 1 for sigmoid
    x3 = tf.reshape(x2, shape=(-1, 4 * 4 * 256))
    logits = tf.layers.dense(inputs=x3, units=1)

    out = tf.sigmoid(logits)

    return out
