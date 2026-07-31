from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='.')

class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Model paths
eff_model_path = 'models/efficientnet_v2.h5'
inception_model_path = 'models/inception_v3.h5'

# Load models with compile=False to avoid optimizer warnings during inference
eff_model = None
inception_model = None

try:
    if os.path.exists(eff_model_path):
        eff_model = load_model(eff_model_path, compile=False)
        print(f"[OK] EfficientNetV2 model loaded successfully")
    else:
        print(f"[WARNING] EfficientNetV2 model not found at {eff_model_path}")
except Exception as e:
    print(f"[ERROR] Failed to load EfficientNetV2 model: {e}")

try:
    if os.path.exists(inception_model_path):
        inception_model = load_model(inception_model_path, compile=False)
        print(f"[OK] InceptionV3 model loaded successfully")
    else:
        print(f"[WARNING] InceptionV3 model not found at {inception_model_path}")
except Exception as e:
    print(f"[ERROR] Failed to load InceptionV3 model: {e}")


def allowed_file(filename):
    """Check if uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_tumor(image_path, model_choice):
    """
    Predict brain tumor type from MRI image.
    
    Both models were trained with raw [0-255] pixel values:
    - EfficientNetV2: Has built-in preprocessing in the base model
    - InceptionV3: Has a Rescaling(1/127.5, offset=-1) layer built into the saved model
    
    So we pass raw [0-255] images to both models.
    """
    IMAGE_SIZE = 224  # Both models were trained with 224x224

    try:
        img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img_array = img_to_array(img)  # Returns float32 array with values [0, 255]
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        return f"Error loading image: {str(e)}", 0.0

    selected_model = None
    predictions = None

    if model_choice == 'efficientnet':
        if eff_model is None:
            return "EfficientNetV2 model is not available", 0.0
        selected_model = eff_model
        predictions = selected_model.predict(img_array, verbose=0)

    elif model_choice == 'inception':
        if inception_model is None:
            return "InceptionV3 model is not available", 0.0
        selected_model = inception_model
        predictions = selected_model.predict(img_array, verbose=0)

    elif model_choice == 'ensemble':
        if eff_model is None and inception_model is None:
            return "No models are available", 0.0
        elif eff_model is not None and inception_model is not None:
            p1 = eff_model.predict(img_array, verbose=0)
            p2 = inception_model.predict(img_array, verbose=0)
            predictions = (p1 + p2) / 2.0
        else:
            # Only one model available, use it
            available_model = eff_model if eff_model is not None else inception_model
            predictions = available_model.predict(img_array, verbose=0)
    else:
        return "Invalid model choice", 0.0

    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = float(np.max(predictions, axis=1)[0])
    predicted_class = class_labels[predicted_class_index]

    if predicted_class == 'notumor':
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {predicted_class.capitalize()}", confidence_score


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if file is in the request
        if 'file' not in request.files:
            return render_template('index.html', result=None, error="No file uploaded. Please select an MRI scan image.")

        file = request.files['file']

        # Check if a file was actually selected
        if file.filename == '':
            return render_template('index.html', result=None, error="No file selected. Please choose an MRI scan image.")

        # Validate file extension
        if not allowed_file(file.filename):
            return render_template('index.html', result=None, 
                                   error=f"Invalid file format. Supported formats: {', '.join(ALLOWED_EXTENSIONS).upper()}")

        model_choice = request.form.get('model', 'ensemble')

        try:
            filename = secure_filename(file.filename)
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_location)

            result, confidence = predict_tumor(file_location, model_choice)
            return render_template('index.html', 
                                   result=result, 
                                   confidence=f"{confidence*100:.2f}", 
                                   file_path=f'/uploads/{filename}',
                                   selected_model=model_choice)
        except Exception as e:
            return render_template('index.html', result=None, error=f"An error occurred: {str(e)}")

    return render_template('index.html', result=None)


@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  Brain Tumor Detection System")
    print("  Server starting at: http://127.0.0.1:5000")
    print("="*60)
    print(f"  EfficientNetV2: {'Loaded' if eff_model else 'Not Available'}")
    print(f"  InceptionV3:    {'Loaded' if inception_model else 'Not Available'}")
    print("="*60 + "\n")
    app.run(debug=True)
