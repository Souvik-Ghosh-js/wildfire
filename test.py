import cv2
import numpy as np
import keras.models

#
# import tensorflow as tf
#
# model = tf.keras.models.load_model(r"C:\Users\91629\Desktop\client\NASAhackathon\NASAproj\NASA\models\Fire.h5")

import tensorflow as tf

# Load the TFLite model
import tensorflow as tf

import tensorflow as tf

# Path to the directory containing the TFLite model
tflite_model_dir = 'quantized_model.tflite'

# Load the TFLite model using tf.lite.TFLiteConverter.from_saved_model()
converter = tf.lite.TFLiteConverter.from_saved_model(tflite_model_dir)

# Convert the TFLite model to a Keras model
keras_model = converter.convert()

# Save the Keras model as an HDF5 (.h5) file
h5_model_path = 'converted_model.h5'
with tf.io.gfile.GFile(h5_model_path, 'wb') as f:
    f.write(keras_model)

