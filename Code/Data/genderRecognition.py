from .DataStrategy import DataStrategy
import numpy as np
import cv2
import h5py

class genderRecognition(DataStrategy):

    def processData(self, direction, gender, split):
        trainX = []
        trainY = []
        testX = []
        testY = []
        classes_array = [b'Woman', b'Man']

        for i in range(len(direction)):
            image = cv2.imread(direction[i])
            gender_current = gender[i]
            split_current = split[i]
            
            if split_current == 'train':
                trainX.append(cv2.resize(cv2.imread(direction[i]), (64, 64), interpolation=cv2.INTER_AREA))
                if gender_current == 'woman':
                    trainY.append(0)
                else:
                    trainY.append(1)
            else:
                testX.append(cv2.resize(cv2.imread(direction[i]), (64, 64), interpolation=cv2.INTER_AREA))
                if gender_current == 'woman':
                    testY.append(0)
                else:
                    testY.append(1)        

        trainX_array = np.array(trainX)
        trainY_array = np.array(trainY)
        testX_array = np.array(testX)
        testY_array = np.array(testY)
        classes_array = np.array(classes_array, dtype='|S7')
        
        return trainX_array, trainY_array, testX_array, testY_array, classes_array

    def saveh5py(self, trainXRandomizing, trainYRandomizing, testXRandomizing, testYRandomizing, classes_array, name):
        with h5py.File(f'./DataSet/{name}.h5', 'w') as hdf_file:
            hdf_file.create_dataset('trainX', data=trainXRandomizing)
            hdf_file.create_dataset('trainY', data=trainYRandomizing)
            hdf_file.create_dataset('testX', data=testXRandomizing)
            hdf_file.create_dataset('testY', data=testYRandomizing)
            hdf_file.create_dataset('classes', data=classes_array)
        return name
