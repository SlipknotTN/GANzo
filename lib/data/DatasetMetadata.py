import json


class DatasetMetadata(object):

    def __init__(self, trainingSamplesNumber=None, numClasses=0):

        self.trainingSamplesNumber = trainingSamplesNumber
        self.numClasses = numClasses

    def initFromJson(self, jsonFile):

        with open(jsonFile, 'r') as jsonDatasetMetadata:
            datasetMetadataDict = json.load(jsonDatasetMetadata)
            self.trainingSamplesNumber = int(datasetMetadataDict["trainingSamplesNumber"])
            self.numClasses = int(datasetMetadataDict["numClasses"])
        return self
