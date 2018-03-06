import lib.model.simpleMNIST as simpleMNIST
import lib.model.dcgan32 as dcgan32


class ModelFactory(object):

    @classmethod
    def create(cls, config):

        # Choose model network and build trainable layers
        if config.architecture.lower() == "simple_mnist":
            return simpleMNIST.generator, simpleMNIST.discriminator
        elif config.architecture.lower() == "dcgan_32":
            return dcgan32.generator, dcgan32.discriminator
        else:
            raise Exception('Architecture ' + config.model + 'not supported')