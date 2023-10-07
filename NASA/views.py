from django.shortcuts import render, redirect


def index(request):
    return render(request , "index.html")

import tensorflow as tf
import numpy as np

# Load the TFLite model
tflite_model_path = 'quantized_model.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define your label mapping
label_mapping = ['No Wild Fire', 'Wild Fire']

def preprocess_image(image_data):
    # Preprocess the image for your TFLite model
    # Example preprocessing assuming your model expects 224x224 images:
    img = tf.image.decode_image(image_data, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict(request):
    if request.method == "POST":
        # Check if the 'upload-img' field exists in the request.FILES dictionary
        if 'upload_img' in request.FILES:
            # Get the uploaded image from request.FILES
            uploaded_image = request.FILES['upload_img']

            # Preprocess the image
            image_data = uploaded_image.read()
            preprocessed_img = preprocess_image(image_data)

            # Perform inference using the TFLite model
            interpreter.set_tensor(input_details[0]['index'], preprocessed_img)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])

            # Post-process the predictions
            max_probability = np.max(predictions)
            predicted_class_index = np.argmax(predictions)
            predicted_class = label_mapping[predicted_class_index]
            accuracy = max_probability * 100

            # Format the accuracy to two decimal places
            formatted_accuracy = "{:.2f}".format(accuracy)

            return render(request, 'index.html', {'predictions': predicted_class, 'accuracy': formatted_accuracy})
        else:
            print("No image was given")
            return render(request, 'index.html', {'error': 'No image was given'})
    else:
        print("No image was given")
        return render(request, 'index.html', {'error': 'No image was given'})
