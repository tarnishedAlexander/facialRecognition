import numpy as np
import pandas as pd
import h5py

class ProcessingData:
    def permutation(self, trainX_array, trainY_array, testX_array, testY_array):
        varPermutationTrain = np.random.permutation(len(trainX_array))
        varPermutationTest = np.random.permutation(len(testX_array))

        trainXRandomizing = trainX_array[varPermutationTrain]
        trainYRandomizing = trainY_array[varPermutationTrain]
        testXRandomizing = testX_array[varPermutationTest]
        testYRandomizing = testY_array[varPermutationTest]

        return trainXRandomizing, trainYRandomizing, testXRandomizing, testYRandomizing

    def loadDataFromCSV(self, csv_file):
        dataframe = pd.read_csv(csv_file)
        
        direction = np.array("./DataSet/facial/" + dataframe['file'])
        gender = np.array(dataframe['gender'])
        split = np.array(dataframe['split'])
        
        return direction, gender, split

    def loadData(self, nameH5py):
        trainDataSet = h5py.File(f'./DataSet/{nameH5py}.h5', 'r')
        
        trainSetX = np.array(trainDataSet["trainX"][:])
        trainSetY = np.array(trainDataSet["trainY"][:])
        testSetX = np.array(trainDataSet["testX"][:])
        testSetY = np.array(trainDataSet["testY"][:])
        classes = np.array(trainDataSet["classes"][:], dtype='|S7')

        trainSetY = trainSetY.reshape((1, trainSetY.shape[0]))
        testSetY = testSetY.reshape((1, testSetY.shape[0]))

        return trainSetX, trainSetY, testSetX, testSetY, classes
