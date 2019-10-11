# notebook follows this tutorial: https://www.tensorflow.org/tutorials/keras/classification
import keras


fashion_mnist = keras.datasets.fashion_mnist
import numpy as np
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# class names are not stored in the dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


train_images = train_images / 255.0

test_images = test_images / 255.0


# building the model consists of configuring the layers and then compiling the model

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), #  transforms the format of the images from a two-dimensional array (of 28 by 28 pixels) to a one-dimensional array (of 28 * 28 = 784 pixels). This layer has no parameters to learn; it only reformats the data.
    keras.layers.Dense(128, activation='relu'), # The first Dense layer has 128 nodes (or neurons)
    keras.layers.Dense(10, activation='softmax') # The second (and last) layer is a 10-node softmax layer that returns an array of 10 probability scores that sum to 1.
])



model.compile(optimizer='adam', #Loss function —This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
              loss='sparse_categorical_crossentropy', # Optimizer —This is how the model is updated based on the data it sees and its loss function.
              metrics=['accuracy']) # Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.

model.fit(train_images, train_labels, epochs=10) # jupyter notebook does not work sometimes as CUDA something something....


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)


predictions = model.predict(test_images)

predictions[0] # spits out ten labels from 0 to 1 with the highest number being the guess.
np.argmax(predictions[0]) # picks the highest one of the set
