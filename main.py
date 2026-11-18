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

# Candidate paths for models (supports newly trained .keras models as well as legacy .h5)
eff_model_candidates = [
    'models/efficientnet_v2_best.keras',
    'models/efficientnet_v2_final.keras',
    'models/efficientnet_v2.keras',
    'models/efficientnet_v2.h5'
]

inception_model_candidates = [
    'models/inception_v3_best.keras',
    'models/inception_v3_final.keras',
    'models/inception_v3.keras',
    'models/inception_v3.h5'
]

eff_model = None
eff_model_path = None
for candidate in eff_model_candidates:
    if os.path.exists(candidate):
        try:
            eff_model = load_model(candidate, compile=False)
            eff_model_path = candidate
            print(f"[OK] EfficientNetV2 model loaded successfully from {candidate}")
            break
        except Exception as e:
            print(f"[WARNING] Failed to load {candidate}: {e}")

if eff_model is None:
    print("[WARNING] No EfficientNetV2 model found.")

inception_model = None
inception_model_path = None
for candidate in inception_model_candidates:
    if os.path.exists(candidate):
        try:
            inception_model = load_model(candidate, compile=False)
            inception_model_path = candidate
            print(f"[OK] InceptionV3 model loaded successfully from {candidate}")
            break
        except Exception as e:
            print(f"[WARNING] Failed to load {candidate}: {e}")

if inception_model is None:
    print("[WARNING] No InceptionV3 model found.")


def get_model_input_size(model, default_size=224):
    """Dynamically determine the expected input resolution (H, W) of the model."""
    if model is None:
        return (default_size, default_size)
    try:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        if shape and len(shape) >= 3 and shape[1] is not None and shape[2] is not None:
            return (int(shape[1]), int(shape[2]))
    except Exception:
        pass
    return (default_size, default_size)


def prepare_image(image_path, target_size):
    """Load and prepare image for inference with appropriate dimensions."""
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)  # values in [0, 255]
    return np.expand_dims(img_array, axis=0)


def allowed_file(filename):
    """Check if uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_tumor(image_path, model_choice):
    """
    Predict brain tumor type from MRI image.
    Handles dynamic input sizes per model (e.g. 300x300 for EfficientNetV2, 299x299 for InceptionV3).
    """
    try:
        eff_size = get_model_input_size(eff_model, default_size=300)
        inc_size = get_model_input_size(inception_model, default_size=299)
    except Exception as e:
        return f"Error determining model shapes: {str(e)}", 0.0

    predictions = None

    if model_choice == 'efficientnet':
        if eff_model is None:
            return "EfficientNetV2 model is not available", 0.0
        try:
            img_array = prepare_image(image_path, eff_size)
            predictions = eff_model.predict(img_array, verbose=0)
        except Exception as e:
            return f"Error during EfficientNet prediction: {str(e)}", 0.0

    elif model_choice == 'inception':
        if inception_model is None:
            return "InceptionV3 model is not available", 0.0
        try:
            img_array = prepare_image(image_path, inc_size)
            predictions = inception_model.predict(img_array, verbose=0)
        except Exception as e:
            return f"Error during Inception prediction: {str(e)}", 0.0

    elif model_choice == 'ensemble':
        if eff_model is None and inception_model is None:
            return "No models are available", 0.0
        try:
            if eff_model is not None and inception_model is not None:
                img_eff = prepare_image(image_path, eff_size)
                img_inc = prepare_image(image_path, inc_size)
                p1 = eff_model.predict(img_eff, verbose=0)
                p2 = inception_model.predict(img_inc, verbose=0)
                predictions = (p1 + p2) / 2.0
            elif eff_model is not None:
                img_eff = prepare_image(image_path, eff_size)
                predictions = eff_model.predict(img_eff, verbose=0)
            else:
                img_inc = prepare_image(image_path, inc_size)
                predictions = inception_model.predict(img_inc, verbose=0)
        except Exception as e:
            return f"Error during Ensemble prediction: {str(e)}", 0.0
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
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "="*60)
    print("  Brain Tumor Detection System")
    print(f"  Server starting at: http://0.0.0.0:{port}")
    print("="*60)
    print(f"  EfficientNetV2: {'Loaded' if eff_model else 'Not Available'}")
    print(f"  InceptionV3:    {'Loaded' if inception_model else 'Not Available'}")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False)
