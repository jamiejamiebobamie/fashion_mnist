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

print(predictions[0]) # spits out ten labels from 0 to 1 with the highest number being the guess.
print(np.argmax(predictions[0])) # picks the highest one of the set

print(test_labels[0])

# copy and paste the below functions to notebook to check out how everything was...
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
# plt.show()


# Grab an image from the test dataset.
img = test_images[1]
print(img.shape)


# tf.keras models are optimized to make predictions on a batch, or collection, of examples at once.
# Accordingly, even though you're using a single image, you need to add it to a list.

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)

predictions_single = model.predict(img)
print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

print(np.argmax(predictions_single[0]))
