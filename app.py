from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model/plant_model.h5")

# Load class labels
with open("model/classes.json", "r") as f:
    class_indices = json.load(f)

# Fix class order
classes = [None] * len(class_indices)
for key, value in class_indices.items():
    classes[value] = key


# 🌍 LANGUAGE DICTIONARY
translations = {
    "en": {
        "upload": "Upload Leaf",
        "scan": "Scan Leaf",
        "diagnosis": "Diagnosis",
        "confidence": "Confidence",
        "healthy": "Healthy Plant",
        "diseased_label": "Diseased",
        "disease_detected": "Disease Detected",
        "click_upload": "Click to upload",
        "or_drag": "or drag & drop",
        "clear_photo": "a clear photo of the leaf",
    },
    "hi": {
        "upload": "पत्ता अपलोड करें",
        "scan": "स्कैन करें",
        "diagnosis": "निदान",
        "confidence": "विश्वास स्तर",
        "healthy": "स्वस्थ पौधा",
        "diseased_label": "रोगग्रस्त",
        "disease_detected": "रोग पाया गया",
        "click_upload": "अपलोड करने के लिए क्लिक करें",
        "or_drag": "या ड्रैग और ड्रॉप करें",
        "clear_photo": "पत्ते की साफ तस्वीर अपलोड करें",
    }
}


# 🔥 Prediction function
def predict_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    class_index = np.argmax(prediction)
    confidence = round(np.max(prediction) * 100, 2)

    return classes[class_index], confidence


# 🟢 MAIN ROUTE
@app.route('/', methods=['GET', 'POST'])
def index():
    lang = request.args.get("lang", "en")   # default English
    t = translations.get(lang, translations["en"])

    result = ""
    confidence = 0
    suggestion = ""

    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', predictions=[], t=t, lang=lang)

        file = request.files['image']
        img = Image.open(file).convert('RGB')

        result, confidence = predict_image(img)

        # Language-based suggestion
        if "healthy" in result.lower():
            suggestion = t["healthy"]
        else:
            suggestion = t.get("diseased", "Disease detected")

    prediction_data = []

    if result:
        # Clean label
        clean_label = result.split("_", 1)[-1]
        clean_label = clean_label.replace("_", " ").capitalize()

        prediction_data = [{
            "label": clean_label,
            "confidence": confidence,
            "severity": "None" if "healthy" in result.lower() else "Moderate",
            "action": suggestion,
            "prevention": "Keep plant in good condition",
            "color": "#2d6a35" if "healthy" in result.lower() else "#c0392b",
            "low_conf": confidence < 50
        }]

    return render_template(
        'index.html',
        predictions=prediction_data,
        t=t,
        lang=lang
    )


# 🔥 API ROUTE
@app.route('/predict', methods=['POST'])
def predict():
    lang = request.args.get("lang", "en")
    t = translations.get(lang, translations["en"])

    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['image']
    img = Image.open(file).convert('RGB')

    result, confidence = predict_image(img)

    clean_label = result.split("_", 1)[-1]
    clean_label = clean_label.replace("_", " ").capitalize()

    if "healthy" in result.lower():
        suggestion = t.get("healthy", "Plant is healthy")
    else:
        suggestion = t.get("diseased", "Disease detected")

    return jsonify({
        "result": clean_label,
        "confidence": confidence,
        "suggestion": suggestion
    })


# 🚀 RUN
if __name__ == "__main__":
    app.run(debug=True)