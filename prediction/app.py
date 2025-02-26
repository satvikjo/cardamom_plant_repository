#necessary libraries
import tensorflow as tf
from tensorflow.keras import utils
from PIL import Image
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import os

model = tf.keras.models.load_model(r'C:\Users\satvik\Documents\Deep Learning Project\best_model.h5')

def preprocess_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.resnet_v2.preprocess_input(img_array)

def predict_disease(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    return np.argmax(prediction)

app = Flask(__name__, template_folder='code')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        # Save the file to disk
        filename = file.filename
        file.save(filename)
        # Predict the disease
        prediction = predict_disease(filename)
        # Delete the file from disk
        os.remove(filename)
        # Return the prediction
        if prediction == 0:
            return "The leaf is infected with blight"
        elif prediction == 2:
            return "The leaf has a leaf spot"
        else:
            return "The leaf is healthy"
    else:
        return render_template('index.html')
    
if __name__ == '__main__':
    app.run(debug=True)