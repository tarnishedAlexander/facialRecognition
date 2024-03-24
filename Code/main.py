from Data.ProcessingData import ProcessingData
from backend.NeuralNetwork import NeuralNetwork
from Data.genderRecognition import genderRecognition
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

processDatas = ProcessingData()
genderData = genderRecognition()

facialRecognise = "./DataSet/facial/gender_detection.csv"
nameH5py = "data"
direction, gender, split = processDatas.loadDataFromCSV(facialRecognise)

trainX_array, trainY_array, testX_array, testY_array, classes_array = genderData.processData(direction, gender, split)

trainX_randomizing, trainY_randomizing, testX_randomizing, testY_randomizing = processDatas.permutation(trainX_array, trainY_array, testX_array, testY_array)

trainSetX, trainSetY, testSetX, testSetY, classes = processDatas.loadData(nameH5py)

nameSaver = genderData.saveh5py(trainX_randomizing, trainY_randomizing, testX_randomizing, testY_randomizing, classes, nameH5py)


m_train = trainSetX.shape[0]
num_px = trainSetX.shape[1]
m_test = testSetX.shape[0]


train_x_flatten = trainSetX.reshape(trainSetX.shape[0], -1).T
test_x_flatten = testSetX.reshape(testSetX.shape[0], -1).T

train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.


layers_dims = [12288,25,5, 1]   
nN = NeuralNetwork(layers_dims)

parameters, cost = nN.LLayer_model(train_x,trainSetY, layers_dims)

my_image = "download.jpeg" 
my_label_y = [1] 

fname = "./DataSet/" + my_image
image = np.array(Image.open(fname).resize((num_px, num_px)))
plt.imshow(image)
image = image / 255.
image = image.reshape((1, num_px * num_px * 3)).T

my_predicted_image = nN.predict(image, my_label_y, parameters)

print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")