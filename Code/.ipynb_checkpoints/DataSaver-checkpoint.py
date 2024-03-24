class DataSaver:
    def save(self, trainX, trainY, testX, testY, classes, file_path):
        raise NotImplementedError("Subclasses must implement save method")