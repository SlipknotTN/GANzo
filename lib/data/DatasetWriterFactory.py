from .DatasetTFWriter import DatasetTFWriter


class DatasetWriterFactory(object):

    @classmethod
    def createDatasetWriter(cls, scriptArgs):

        # Init Dataset TF Writer
        dataset = DatasetTFWriter()
        dataset.setTrainValSamplesList(imagesDir=scriptArgs.imagesDir)

        return dataset
