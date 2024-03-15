import os; os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
import numpy as np
from PIL import Image
from keras.layers import Dense, Flatten
from keras.models import Sequential


class NeuralNetwork:
    def __init__(self) -> None:
        self.model = None

    def setModel(self) -> None:
        self.model = Sequential()
        self.model.add(Dense(32, activation='relu', input_shape=(28, 28)))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(10, activation='sigmoid'))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    def trainModel(self) -> None:
        from keras.datasets import mnist
        (x_train, y_train) = mnist.load_data()[0]
        x_train = x_train / 255
        y_train = keras.utils.to_categorical(y_train, 10)
        self.model.fit(x_train, y_train, epochs=50)
        self.model.save('model.h5')

    def loadModel(self) -> None:
        self.model = keras.models.load_model('model.h5')

    def predict(self, image) -> None:
        img_array = (255 - np.asarray(Image.open(image).convert('L'))) / 255
        predict = ((self.model.predict(np.array([img_array]))).tolist())[0]
        print(predict.index(max(predict)))


if __name__ == '__main__':
    neural_network = NeuralNetwork()
    neural_network.loadModel()
    neural_network.predict('image.png')
