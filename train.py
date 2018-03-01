import argparse
import os

from skimage.color import gray2rgb
from skimage.io import imsave
import numpy as np
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

    if not tf.gfile.Exists(args.datasetDir):
        tf.gfile.MakeDirs(args.datasetDir)

    download_and_convert_mnist.run(args.datasetDir)

    # Set up the input
    originalImages, labels, numSamples = mnist_data_provider.provide_data("train", config.batchSize, args.datasetDir)
    images = tf.reshape(originalImages, shape=(config.batchSize, originalImages.shape[1] * originalImages.shape[2]))
    noise = tf.random_normal([config.batchSize, config.noiseSize])

    # Build the generator and discriminator.
    gan_model = tfgan.gan_model(
        generator_fn=simpleMNIST.generator,  # you define
        discriminator_fn=simpleMNIST.discriminator,  # you define
        real_data=images,
        generator_inputs=noise)

    # Build the GAN loss.
    gan_loss = tfgan.gan_loss(
        gan_model,
        generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
        discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss)

    # Create the train ops, which calculate gradients and apply updates to weights.
    train_ops = tfgan.gan_train_ops(
        gan_model,
        gan_loss,
        generator_optimizer=tf.train.AdamOptimizer(0.5),
        discriminator_optimizer=tf.train.AdamOptimizer(0.5))

    # Run the train ops in the alternating training scheme.
    tfgan.gan_train(
        train_ops,
        hooks=[tf.train.StopAtStepHook(num_steps=10000)],
        logdir=args.tensorboardDir)

    # Save images
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        generatedData = sess.run(gan_model.generated_data)
        generatedData = np.reshape(generatedData, newshape=originalImages.shape)
        splittedImages = np.split(generatedData, indices_or_sections=generatedData.shape[0], axis=0)
        if os.path.exists(os.path.dirname(args.outputImagePath)) is False:
            os.makedirs(os.path.dirname(args.outputImagePath))
        # Squeeze first dimension to have 3D numpy array with clip to -1 and 1 in case of strange predictions
        for index, image in enumerate(splittedImages):
            filePath = args.outputImagePath + "_" + str(index + 1) + ".jpg"
            image = np.squeeze(image, axis=0)
            image = np.squeeze(image, axis=-1)
            imageRGB = gray2rgb(image)
            imsave(filePath, np.clip(imageRGB, a_min=-1.0, a_max=1.0))
            print("Saved sample in " + filePath)


if __name__ == "__main__":
    main()