from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io, os

app = Flask(__name__)

# Load the pre-trained model
model = load_model('model/mnist_model.h5')
  # For TensorFlow-based model

# For scratch-built model, load weights (uncomment if using)
# data = np.load('mnist_nn_weights.npz')
# w1, w2, b1, b2 = data['w1'], data['w2'], data['b1'], data['b2']

UPLOAD_FOLDER = 'static/uploads'  # Make sure this folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image)  # Convert to numpy array
    image = image.reshape(1, 28 * 28)  # Reshape to (1, 784) for model input
    image = image / 255.0  # Normalize the pixel values
    return image


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            img = Image.open(io.BytesIO(file.read()))
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            img.save(img_path)
            img1 = preprocess_image(img)
            prediction = np.argmax(model.predict(img1))  # For TensorFlow model

            # For scratch-built model, use the weights and make predictions:
            # prediction = custom_nn_predict(img, w1, w2, b1, b2)

            return render_template("result.html", prediction=prediction, image = url_for('static', filename='uploads/' + file.filename))
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
