from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import PIL.Image

# Flask app initialization
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploaded'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load the TF Hub style transfer model once
print("🔄 Loading model...")
style_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
print("✅ Model loaded.")

# Helper to load image
def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (512, 512))
    img = img[tf.newaxis, :]
    return img

# Helper to convert tensor to image
def deprocess_img(img):
    img = img[0]
    img = tf.clip_by_value(img, 0.0, 1.0)
    img = (img.numpy() * 255).astype(np.uint8)
    return PIL.Image.fromarray(img)

# Route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        content_file = request.files['content']
        style_file = request.files['style']

        if content_file and style_file:
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(content_file.filename))
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(style_file.filename))
            content_file.save(content_path)
            style_file.save(style_path)

            content_image = load_img(content_path)
            style_image = load_img(style_path)

            stylized = style_model(tf.constant(content_image), tf.constant(style_image))[0]
            result_img = deprocess_img(stylized)

            result_path = os.path.join(app.config['RESULT_FOLDER'], 'result.jpg')
            result_img.save(result_path)

            return render_template('index.html', result_image=result_path)

    return render_template('index.html', result_image=None)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
