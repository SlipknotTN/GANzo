import argparse

import tensorflow as tf
tfgan = tf.contrib.gan

from lib.config.ConfigParams import ConfigParams

def doParsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='TF GAN API training scripts')
    #parser.add_argument('--datasetDir', required=True, default=None, help='Dataset directory')
    parser.add_argument('--modelOutputDir', required=False, default="./export",
                        help='Output folder that will contains final trained model graph.pb')
    parser.add_argument('--configFile', required=True, help='Config File for training')
    parser.add_argument('--tensorboardDir', required=False, default=None, help="TensorBoard directory")
    parser.add_argument('--useGpu', type=str, required=False, default=None,
                        help='GPU ID to use for the training (default CPU training)')
    return parser.parse_args()


def main():

    args = doParsing()
    print(args)

    config = ConfigParams(args.configFile)

    # Set up the input.
    images = mnist_data_provider.provide_data(config.batchSize)
    noise = tf.random_normal([config.batchSize, args.noiseSize])

    # Build the generator and discriminator.
    gan_model = tfgan.gan_model(
        generator_fn=mnist.unconditional_generator,  # you define
        discriminator_fn=mnist.unconditional_discriminator,  # you define
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
        hooks=[tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps)],
        logdir=FLAGS.train_log_dir)



if __name__ == "__main__":
    main()