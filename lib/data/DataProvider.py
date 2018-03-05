import tensorflow as tf

import lib.data.mnist.data_provider as mnist_data_provider

from deps.tfmodels.research.slim.datasets import download_and_convert_mnist


class DataProvider(object):

    @classmethod
    def createDataProvider(cls, scriptArgs, config):

        if scriptArgs.datasetName == "mnist":
            if not tf.gfile.Exists(scriptArgs.datasetDir):
                tf.gfile.MakeDirs(scriptArgs.datasetDir)
            download_and_convert_mnist.run(scriptArgs.datasetDir)
            # Set up the input
            originalImages, labels, numSamples = mnist_data_provider.provide_data("train", config.batchSize,
                                                                                  scriptArgs.datasetDir)
            # Range is [-1, 1]
            images = tf.reshape(originalImages,
                                shape=(config.batchSize, originalImages.shape[1] * originalImages.shape[2]))
            return images, numSamples

        elif scriptArgs.datasetName == "custom":

            raise NotImplementedError

        else:
            raise Exception("Dataset " +  scriptArgs.datasetName + " not supported")