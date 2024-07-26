import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.models import Sequential

num_classes = 8
img_height = 180
img_width = 180
batch_size = 36

class_names = [
    'Levetiracetam 500 Tablet',
    'Ibruwell 400 Tablet',
    'Alprozolam Tablet',
    'Tripolidine Tablet',
    'Cephalexin Capsule',
    'Thera-Tabs Tablet',
    'Folivane Capsule',
    'Dolo 650-MG'
    ]

print(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.load_weights('pill_model.h5')


def detect_pill_from_img(image_file):
  print(f'Your name is: {image_file}')
  img = tf.keras.utils.load_img(
      image_file, target_size=(img_height, img_width)
  )

  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])
  print(
      "This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score)], 100 * np.max(score))
  )
  return class_names[np.argmax(score)],100 * np.max(score)