""" The MNIST dataset expects 28x28 grayscale values
 scaled between 0.0 (black background)and 1.0 (white digit)
"""
from flask import Flask, request, render_template, jsonify
from model import make_predictions
from PIL import Image #python imaging library
import numpy as np  
import base64
import io

app = Flask(__name__)

data = np.load("models/mnist_parameters.npz")
W1 = data["W1"]
b1 = data["b1"]
W2 = data["W2"]
b2 = data["b2"]

def process_image(image):
    image = Image.open(io.BytesIO(image)).convert("L") # convert to grayscale
    image = image.resize((28, 28))
    image = np.array(image)
    image = image / 255 # normalize image to [0, 1]
    image = image.reshape(784, 1)

    # print("shape:", image.shape)
    # print("min:", image.min())
    # print("max:", image.max())
    # print("mean:", image.mean())
    # print("non-zero pixels:", np.count_nonzero(image))

    return image 
    # returns a flattened 784 pixels grayscale image array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])

def predict():
    data = request.get_json()
    image_data = data.get('image', '')

    # strips the meta data header
    if "," in image_data:
        image_data = image_data.split(',')[1]

    # decode base64 string to binary bytes
    image_bytes = base64.b64decode(image_data)

    image = process_image(image_bytes)

    prediction = make_predictions(
        image,
        W1,
        b1,
        W2, 
        b2)
    
    return jsonify({
        "prediction": int(prediction[0])
    })

if __name__ == "__main__":
    app.run()
