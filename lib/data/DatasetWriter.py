import glob
import os
import random


class DatasetWriter(object):

    def __init__(self):

        self.numClasses = None
        self.trainingSamplesList = None

    def setTrainValSamplesList(self, imagesDir):
        """
        Set images file list with training samples.
        Root directory must contains images files or subdirectories with single classes images.
        :param imagesDir:
        :return: list of tuples filename, classIndex
        """
        tuplesList = []

        # Check if the directory contains classes subdirectories -> Class loss
        classNames = [f for f in os.listdir(imagesDir) if os.path.isdir(os.path.join(imagesDir, f))]
        if classNames != list():
            classNames = sorted(classNames)
            for classIndex, className in enumerate(classNames):
                print(classIndex, className)
                files = sorted(glob.glob(os.path.join(imagesDir, className) + "/*.*"), key=lambda s: s.lower())
                for file in files:
                    tuplesList.append((file, classIndex))
        else:
            # Unsupervised training
            files = sorted(glob.glob(imagesDir + "/*.*"), key=lambda s: s.lower())
            for file in files:
                tuplesList.append((file, 0))

        random.shuffle(tuplesList)
        self.numClasses = len(classNames)
        self.trainingSamplesList = tuplesList

    def getTrainingSamplesNumber(self):

        return len(self.trainingSamplesList)
