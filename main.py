import os; os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
import numpy as np
from PIL import Image
from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D


class NeuralNetwork:
    def __init__(self) -> None:
        self.model = None

    def setModel(self) -> None:
        self.model = Sequential([
            Conv2D(8, 3, input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=2),
            Flatten(),
            Dense(10, activation='softmax'),
        ])

        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def trainModel(self) -> None:
        from keras.datasets import mnist
        (x_train, y_train) = mnist.load_data()[0]
        x_train = x_train / 255 - 0.5
        x_train = np.expand_dims(x_train, axis=3)
        y_train = keras.utils.to_categorical(y_train, 10)
        self.model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=1)
        self.model.save('model.h5')

    def loadModel(self) -> None:
        self.model = keras.models.load_model('model.h5')

    def predict(self, image) -> None:
        img_array = (255 - np.asarray(Image.open(image).convert('L'))) / 255 - 0.5
        predict = ((self.model.predict(np.array([img_array]))).tolist())[0]
        print(predict.index(max(predict)))


if __name__ == '__main__':
    neural_network = NeuralNetwork()
    neural_network.loadModel()
    neural_network.predict('image.png')
