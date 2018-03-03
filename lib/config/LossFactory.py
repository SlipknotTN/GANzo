import sys

import tensorflow as tf
tfgan = tf.contrib.gan


class LossFactory(object):

    # Available losses: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/gan/python/losses/python/losses_impl.py

    @classmethod
    def createLoss(cls, lossName, generator):

        if generator:
            suffix = "_generator_loss"
        else:
            suffix = "_discriminator_loss"

        # Direct string -> function getter
        try:
            return getattr(tfgan.losses, lossName + suffix)
        except AttributeError as e:
            raise Exception("Loss " + lossName + " not supported")
