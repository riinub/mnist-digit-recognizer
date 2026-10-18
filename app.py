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

data = np.load("models/mnist_parameters2.npz")
for key in data.files:
    print(key, data[key].shape)

W1 = data["param_0"]
b1 = data["param_1"]

W2 = data["param_2"]
b2 = data["param_3"]

W3 = data['param_4']
b3 = data['param_5']


def process_image(image):
    image = Image.open(io.BytesIO(image)).convert("L")

    # Convert to a NumPy array
    image = np.array(image)

    # If the canvas has a white background and dark strokes,
    # invert it to match MNIST: black background, white digit.
    if image.mean() > 127:
        image = 255 - image

    # Find the digit's bounding box
    mask = image > 30
    rows, cols = np.where(mask)

    if len(rows) == 0:
        return np.zeros((784, 1), dtype=np.float32)

    top, bottom = rows.min(), rows.max() + 1
    left, right = cols.min(), cols.max() + 1

    # Crop around the digit
    image = image[top:bottom, left:right]

    # Preserve proportions while resizing to fit within 20x20
    h, w = image.shape
    scale = 20 / max(h, w)

    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    image = Image.fromarray(image).resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS
    )

    # Center the digit on a 28x28 black canvas
    canvas = Image.new("L", (28, 28), 0)

    x = (28 - new_w) // 2
    y = (28 - new_h) // 2

    canvas.paste(image, (x, y))

    # Normalize and flatten to match TensorFlow's input format
    image = np.array(canvas, dtype=np.float32) / 255.0
    image = image.reshape(784, 1)

    return image


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
        W1, b1, 
        W2, b2, 
        W3, b3)
    
    return jsonify({
        "prediction": int(prediction[0])
    })

if __name__ == "__main__":
    app.run()
