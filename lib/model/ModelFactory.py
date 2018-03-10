import lib.model.simpleMNIST as simpleMNIST
import lib.model.dcgan32 as dcgan32
import lib.model.dcgan64 as dcgan64
import lib.model.dcgan128 as dcgan128
import lib.model.dcgan256 as dcgan256
import lib.model.wgan64 as wgan64


class ModelFactory(object):

    @classmethod
    def create(cls, config):

        # TODO: Exploit reflection to save if else lines ...

        # Choose model network and build trainable layers
        if config.architecture.lower() == "simple_mnist":
            return simpleMNIST.generator, simpleMNIST.discriminator
        elif config.architecture.lower() == "dcgan_32":
            return dcgan32.generator, dcgan32.discriminator
        elif config.architecture.lower() == "dcgan_64":
            return dcgan64.generator, dcgan64.discriminator
        elif config.architecture.lower() == "dcgan_128":
            return dcgan128.generator, dcgan128.discriminator
        elif config.architecture.lower() == "dcgan_256":
            return dcgan256.generator, dcgan256.discriminator
        elif config.architecture.lower() == "wgan_64":
            return wgan64.generator, wgan64.discriminator
        else:
            raise Exception('Architecture ' + config.architecture.lower() + ' not supported')