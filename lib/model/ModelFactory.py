import lib.model.simpleMNIST as simpleMNIST
import lib.model.dcgan32 as dcgan32
import lib.model.dcgan64 as dcgan64


class ModelFactory(object):

    @classmethod
    def create(cls, config):

        # Choose model network and build trainable layers
        if config.architecture.lower() == "simple_mnist":
            return simpleMNIST.generator, simpleMNIST.discriminator
        elif config.architecture.lower() == "dcgan_32":
            return dcgan32.generator, dcgan32.discriminator
        elif config.architecture.lower() == "dcgan_64":
            return dcgan64.generator, dcgan64.discriminator
        else:
            raise Exception('Architecture ' + config.architecture.lower() + ' not supported')