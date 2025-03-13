from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load your trained skin tone model
model = load_model('best_model_final.h5')

# Preprocessing function (adjust as per your model's input shape)
def preprocess_image(image):
    image = image.resize((224, 224))  # Adjust size to match model's expected input
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty file'}), 400

    try:
        img = Image.open(io.BytesIO(file.read()))  # Open directly, no need for io.BytesIO
        img = preprocess_image(img)

        predictions = model.predict(img)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions))

        print(f"Predicted class: {predicted_class}, Confidence: {confidence}")

        predicted_class_name = ""
        if predicted_class == 0:
            predicted_class_name = "Dark"
        elif predicted_class == 1:
            predicted_class_name = "Light"
        elif predicted_class == 2:
            predicted_class_name = "Mid Dark"
        else:
            predicted_class_name = "Mid Light"

        return jsonify({'predicted_class': predicted_class_name, 'confidence': confidence})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
