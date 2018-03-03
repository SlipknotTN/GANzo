import argparse

import tensorflow as tf
tfgan = tf.contrib.gan

from lib.config.ConfigParams import ConfigParams
import lib.model.simpleMNIST as simpleMNIST
import lib.data.mnist.data_provider as mnist_data_provider

from deps.tfmodels.research.slim.datasets import download_and_convert_mnist


def doParsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='TF GAN API training scripts')
    parser.add_argument('--datasetDir', required=True, default=None, help='Dataset directory')
    parser.add_argument('--modelOutputDir', required=False, default="./export",
                        help='Output folder that will contains final trained model graph.pb')
    parser.add_argument('--outputImagePath', required=False, default="./export/image",
                        help='Export images path')
    parser.add_argument('--configFile', required=True, help='Config File for training')
    parser.add_argument('--tensorboardDir', required=False, default=None, help="TensorBoard directory")
    parser.add_argument('--useGpu', type=str, required=False, default=None,
                        help='GPU ID to use for the training (default CPU training)')
    return parser.parse_args()


def main():

    args = doParsing()
    print(args)

    config = ConfigParams(args.configFile)

    # TODO: Create an abstraction level for predefined datasets and custom dataset

    if not tf.gfile.Exists(args.datasetDir):
        tf.gfile.MakeDirs(args.datasetDir)

    download_and_convert_mnist.run(args.datasetDir)

    # Set up the input
    originalImages, labels, numSamples = mnist_data_provider.provide_data("train", config.batchSize, args.datasetDir)
    numSteps = numSamples * config.epochs
    print("NumSteps for " + str(config.epochs) + " epochs: " + str(numSteps))
    # Range is [-1, 1]
    images = tf.reshape(originalImages, shape=(config.batchSize, originalImages.shape[1] * originalImages.shape[2]))
    noise = tf.random_normal([config.batchSize, config.noiseSize])

    # TODO: Debug input images

    # Build the generator and discriminator.
    gan_model = tfgan.gan_model(
        generator_fn=simpleMNIST.generator,  # you define
        discriminator_fn=simpleMNIST.discriminator,  # you define
        real_data=images,
        generator_inputs=noise)

    # Build the GAN loss.
    gan_loss = tfgan.gan_loss(
        gan_model,
        generator_loss_fn=config.gloss,
        discriminator_loss_fn=config.dloss)

    # TODO: Add a log in console about training progress (right now we have to check the tensorboard)

    # Create the train ops, which calculate gradients and apply updates to weights.
    train_ops = tfgan.gan_train_ops(
        gan_model,
        gan_loss,
        generator_optimizer=config.optimizer,
        discriminator_optimizer=config.optimizer)

    status_message = tf.string_join(
        ['Starting train step: ',
         tf.as_string(tf.train.get_or_create_global_step())],
        name='status_message')

    # TODO: Add saving of images to tensorboard to evaluate training results

    # TODO: Add inception and frechet distance,
    # see https://github.com/tensorflow/models/blob/master/research/gan/tutorial.ipynb

    # Run the train ops in the alternating training scheme.
    tfgan.gan_train(
        train_ops,
        hooks=[tf.train.StopAtStepHook(num_steps=numSteps),
               tf.train.LoggingTensorHook([status_message], every_n_iter=10)],
        logdir=args.tensorboardDir)

    # From tutorial (it is suitable for evaluating during training, try to add as hook + tensorboard)
    # The output is still a Tensor and you need a session to evaluate
    with tf.variable_scope('Generator', reuse=True):

        generatedDataTensor = gan_model.generator_fn(
            tf.random_normal([config.batchSize, config.noiseSize]))

    # Single images with table of samples
    generatedDataTensor = tf.reshape(generatedDataTensor, shape=originalImages.shape)
    generated_data_to_visualize = tfgan.eval.image_reshaper(
        generatedDataTensor[:config.batchSize, ...], num_cols=10)


if __name__ == "__main__":
    main()