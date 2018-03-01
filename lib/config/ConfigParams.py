import configparser

from lib.constants.Constants import Constants as const


class ConfigParams(object):

    def __init__(self, file):

        config = configparser.ConfigParser()
        config.read_file(open(file))

        # Model
        self.architecture = config.get(const.ConfigSection.model, "architecture")
        self.noiseSize = config.getint(const.ConfigSection.model, "noiseSize", fallback=100)
        self.inputSize = config.getint(const.ConfigSection.model, "inputSize", fallback=224)
        self.inputChannels = config.getint(const.ConfigSection.model, "inputChannels", fallback=3)
        self.inputFormat = config.get(const.ConfigSection.model, "inputFormat", fallback="RGB")

        # HyperParameters
        self.epochs = config.getint(const.ConfigSection.hyperparameters, "epochs")
        self.batchSize = config.getint(const.ConfigSection.hyperparameters, "batchSize")
