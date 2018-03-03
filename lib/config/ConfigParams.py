import configparser

from lib.constants.Constants import Constants as const
from .OptimizerParamsFactory import OptimizerParamsFactory
from .TrainOptimizerFactory import TrainOptimizerFactory
from .LossFactory import LossFactory


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

        # Load losses function (TFGAN API exposes them)
        self.gloss = LossFactory.createLoss(config.get(const.ConfigSection.hyperparameters, const.TrainConfig.gloss),
                                            generator=True)
        self.dloss = LossFactory.createLoss(config.get(const.ConfigSection.hyperparameters, const.TrainConfig.dloss),
                                            generator=False)

        # TODO: Same optimizer for generator and discriminator with this config,
        # need to change optimizer params read style to support same optimizer
        # but with differente parameters from generator to discrinator


        # In dog_breed repository there is an example on how to use different LR policies,
        # here with use fixed LR + adaptive optimizer

        # Load the optimizer params
        optimizerType = str(config.get(const.ConfigSection.hyperparameters, const.TrainConfig.optimizer)).upper()
        self.optimizerParams = OptimizerParamsFactory.createOptimizerParams(optimizerType=optimizerType, config=config)

        # We have to pass learning rate again to add further support to LR policies
        self.optimizer = TrainOptimizerFactory.createOptimizer(learningRate=self.optimizerParams.learning_rate,
                                                               optimizerParams=self.optimizerParams)
