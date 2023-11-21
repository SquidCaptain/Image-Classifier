import tensorflow as tf

# Load the CIFAR10 dataset
cifar10 = tf.keras.datasets.cifar10
(training_images, training_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize the pixel values
training_images = training_images / 255.0
test_images = test_images / 255.0

# Self defined Callback
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):

    # Halts the training when the loss falls below 0.3
    # Check the loss
    if(logs.get('loss') < 0.3):

      # Stop if threshold is met
      print("\nLoss is lower than 0.4 so cancelling training!")
      self.model.stop_training = True

    # Halts the training when the accuracy goes above 0.9
    # Check the accuracy
    if(logs.get('accuracy') > 0.9):

      # Stop if threshold is met
      print("\nAccuracy is above 0.95 so cancelling training!")
      self.model.stop_training = True

# Instantiate class
callbacks = myCallback()

# Define the model
model = tf.keras.models.Sequential([

  # Add convolution and max poolings
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),

  # Add the neurons
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Print the model summary
model.summary()

# Use same settings
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
print(f'\nMODEL TRAINING:')
model.fit(training_images, training_labels, epochs=10)

# Evaluate on the test set
print(f'\nMODEL EVALUATION:')
test_loss = model.evaluate(test_images, test_labels)

# Using the model to classify
import numpy as np
from google.colab import files
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt

uploaded = files.upload()

for fn in uploaded.keys():

  # predicting images
  path = '/content/' + fn
  img = load_img(path, target_size=(32, 32))
  x = img_to_array(img)
  x /= 255
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=1)
  best_guess = np.argmax(classes[0])

  match best_guess:
    case 0:
      print(fn + " is an airplane")
    case 1:
      print(fn + " is an automobile")
    case 2:
      print(fn + " is a bird")
    case 3:
      print(fn + " is a cat")
    case 4:
      print(fn + " is a deer")
    case 5:
      print(fn + " is a dog")
    case 6:
      print(fn + " is a frog")
    case 7:
      print(fn + " is a horse")
    case 8:
      print(fn + " is a ship")
    case 9:
      print(fn + " is a truck")
