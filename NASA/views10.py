from django.shortcuts import render, redirect


def index(request):
    return render(request , "index.html")

import tensorflow as tf  # or your preferred library for image processing
from tensorflow import keras  # or your preferred library for model loading

model = keras.models.load_model('NASA/models/Fire.h5')
import numpy as np
def predict(request):
    if request.method == "POST":
        # Check if the 'upload-img' field exists in the request.FILES dictionary
        if 'upload_img' in request.FILES:
            # Get the uploaded image from request.FILES
            uploaded_image = request.FILES['upload_img']

            # Preprocess the image (resize, normalize, etc.) for your model
            # Example preprocessing using TensorFlow and assuming your model expects 224x224 images:
            img = tf.image.decode_image(uploaded_image.read(), channels=3)
            img = tf.image.resize(img, (224, 224))
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.expand_dims(img, axis=0)  # Add batch dimension
            label = ['No Wild Fire', 'Wild Fire']
            # Make predictions using your model
            predictions = model.predict(img)
            ac = np.max(predictions)
            acc = float(ac * 100)
            formatted_acc = "{:.2f}".format(acc)

            pred=label[np.argmax(predictions)]

            print(predictions)
            # You can process and format the predictions as needed

            return render(request, 'index.html', {'predictions': pred , 'accuracy': formatted_acc })
        else:
            print("hello")

            return render(request , 'index.html', {'error': 'No image was given'})
    else:
        print("hi")
        return render(request, 'index.html', {'error': 'No image was given'})

