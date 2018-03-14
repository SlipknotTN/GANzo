import tensorflow as tf


def generator(z, out_channel_dim=3, alpha=0.2, keepProb=0.7, isTraining=True):
    """
    Create the generator network
    :param z: Input z
    :param out_channel_dim: The number of channels in the output image
    :param is_train: Boolean if generator is being used for training
    :return: The tensor output of the generator
    """

    # DCGAN 64 architecture

    # 1st block -> 4x4x1024
    # First fully connected layer
    # From input (1D to 1D of 4*4*1024)
    x = tf.layers.dense(z, units=(4 * 4 * 1024))
    # Reshahe the fully connected result (HWC)
    x1 = tf.reshape(x, shape=(-1, 4, 4, 1024))
    # Batch Norm + Leaky ReLU
    x1 = tf.layers.batch_normalization(x1, training=isTraining)
    x1 = tf.maximum(alpha * x1, x1)
    x1 = tf.nn.dropout(x1, keep_prob=keepProb)

    # 2nd block -> 8x8x512
    x2 = tf.layers.conv2d_transpose(x1, filters=512, kernel_size=(2, 2), strides=2,
                                    padding='same', activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
    # Batch Norm + Leaky ReLU
    x2 = tf.layers.batch_normalization(x2, training=isTraining)
    x2 = tf.maximum(alpha * x2, x2)
    x2 = tf.nn.dropout(x2, keep_prob=keepProb)

    # 3rd block -> 16x16x256
    # Strides 2 with padding 'same' give the output 16x16 (for the filters we explicitly set 256)
    x3 = tf.layers.conv2d_transpose(x2, filters=256, kernel_size=(3, 3), strides=2,
                                    padding='same', activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
    # Batch Norm + Leaky ReLU
    x3 = tf.layers.batch_normalization(x3, training=isTraining)
    x3 = tf.maximum(alpha * x3, x3)
    x3 = tf.nn.dropout(x3, keep_prob=keepProb)

    # 4th block -> 32x32x128
    # Strides 2 with padding 'same' give the output 32x32 (for the filters we explicitly set 128)
    x4 = tf.layers.conv2d_transpose(x3, filters=128, kernel_size=(5, 5), strides=2,
                                    padding='same', activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
    x4 = tf.layers.batch_normalization(x4, training=isTraining)
    x4 = tf.maximum(alpha * x4, x4)
    x4 = tf.nn.dropout(x4, keep_prob=keepProb)

    # 5th block -> 64x64x64
    # Strides 2 with padding 'same' give the output 64x64 (for the filters we explicitly set 64)
    x5 = tf.layers.conv2d_transpose(x4, filters=64, kernel_size=(5, 5), strides=2,
                                    padding='same', activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
    x5 = tf.layers.batch_normalization(x5, training=isTraining)
    x5 = tf.maximum(alpha * x5, x5)
    x5 = tf.nn.dropout(x5, keep_prob=keepProb)

    # 6th block -> 128x128x32
    # Strides 2 with padding 'same' give the output 128x128 (for the filters we explicitly set 32)
    x6 = tf.layers.conv2d_transpose(x5, filters=32, kernel_size=(5, 5), strides=2,
                                    padding='same', activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
    x6 = tf.layers.batch_normalization(x6, training=isTraining)
    x6 = tf.maximum(alpha * x6, x6)
    x6 = tf.nn.dropout(x6, keep_prob=keepProb)

    # 7th block -> 256x256x16
    # Strides 2 with padding 'same' give the output 256x256 (for the filters we explicitly set 16)
    x7 = tf.layers.conv2d_transpose(x6, filters=16, kernel_size=(5, 5), strides=2,
                                    padding='same', activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
    x7 = tf.layers.batch_normalization(x7, training=isTraining)
    x7 = tf.maximum(alpha * x7, x7)
    x7 = tf.nn.dropout(x7, keep_prob=keepProb)

    # 8th block -> 512x512xCH (no batch normalization and use tanh activation)
    logits = tf.layers.conv2d_transpose(x7, filters=out_channel_dim, kernel_size=(5, 5), strides=2,
                                        padding='same', activation=None,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())

    # Output layer, 512x512xCH
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

    # 1st block - Input layer is 512x512xCH (3 or 1) -> 256x256x64
    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(3, 3), strides=2, activation=None,
                         padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
    # No batch normalization, better results
    x = tf.maximum(alpha * x, x)
    x = tf.nn.dropout(x, keep_prob=keepProb)

    # 2nd block -> 128x128x128
    x1 = tf.layers.conv2d(x, filters=128, kernel_size=(3, 3), strides=2, activation=None,
                          padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
    x1 = tf.layers.batch_normalization(x1, training=isTraining)
    x1 = tf.maximum(alpha * x1, x1)
    x1 = tf.nn.dropout(x1, keep_prob=keepProb)

    # 3rd block -> 64x64x256
    x2 = tf.layers.conv2d(x1, filters=256, kernel_size=(3, 3), strides=2, activation=None,
                          padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
    x2 = tf.layers.batch_normalization(x2, training=isTraining)
    x2 = tf.maximum(alpha * x2, x2)
    x2 = tf.nn.dropout(x2, keep_prob=keepProb)

    # 4th block -> 32x32x512
    # x3 = tf.layers.conv2d(x2, filters=512, kernel_size=(3, 3), strides=2, activation=None,
    #                       padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
    # x3 = tf.layers.batch_normalization(x3, training=isTraining)
    # x3 = tf.maximum(alpha * x3, x3)
    # x3 = tf.nn.dropout(x3, keep_prob=keepProb)
    #
    # # 5th block - Reshape for final dense layer -> units = 1 for sigmoid
    # x4 = tf.reshape(x3, shape=(-1, 32 * 32 * 512))
    # logits = tf.layers.dense(inputs=x4, units=1)

    # Alternative end #1 at 4th block - Reshape for final dense layer -> units = 1 for sigmoid
    # Better behaviour of this architecture,
    # with 5 layers the discriminator is too "smart" w.r.t. the generator.
    # x3 = tf.reshape(x2, shape=(-1, 64, 64, 256))
    # logits = tf.layers.dense(inputs=x3, units=1)

    # Alternative #2 - Simpler 4th block - Conv 1x1 - Same size 64x64x256
    x3 = tf.layers.conv2d(x2, filters=256, kernel_size=(1, 1), strides=1, activation=None,
                          padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
    x3 = tf.layers.batch_normalization(x3, training=isTraining)
    x3 = tf.maximum(alpha * x3, x3)
    x3 = tf.nn.dropout(x3, keep_prob=keepProb)

    x4 = tf.reshape(x3, shape=(-1, 64, 64, 256))
    logits = tf.layers.dense(inputs=x4, units=1)

    out = tf.sigmoid(logits)

    return out
