from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import gdown

app = Flask(__name__)

# =========================
# CONFIG
# =========================
app.config['SECRET_KEY'] = 'secret123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

db = SQLAlchemy(app)

# =========================
# LOGIN SETUP
# =========================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# =========================
# MODELS
# =========================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    password = db.Column(db.String(100))


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    label = db.Column(db.String(100))
    confidence = db.Column(db.Float)
    image = db.Column(db.String(200))


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# =========================
# AUTH ROUTES
# =========================
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        user = User(
            username=request.form['username'],
            password=request.form['password']
        )
        db.session.add(user)
        db.session.commit()
        return redirect('/login')
    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(
            username=request.form['username'],
            password=request.form['password']
        ).first()

        if user:
            login_user(user)
            return redirect('/')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/login')

# =========================
# LOAD MODEL
# =========================
if not os.path.exists("model/plant_model.h5"):
    url = "https://drive.google.com/uc?id=1ZMu-Ha3KPV_Fb_KtJ2VsSlkffFyeANWW"
    gdown.download(url, "model/plant_model.h5", quiet=False)

model = tf.keras.models.load_model("model/plant_model.h5")

# =========================
# CLASS LABELS
# =========================
with open("model/classes.json", "r") as f:
    class_indices = json.load(f)

classes = [None] * len(class_indices)
for key, value in class_indices.items():
    classes[value] = key

# =========================
# SOLUTIONS
# =========================
solutions = {
    "Healthy": "No treatment needed. Keep monitoring the plant.",
    "Leaf spot": "Remove infected leaves and apply fungicide spray.",
    "Powdery mildew": "Use sulfur spray and avoid overwatering.",
    "Rust": "Apply neem oil or copper-based fungicide.",
    "Blight": "Remove affected parts and use proper pesticides."
}

# =========================
# PREDICTION FUNCTION
# =========================
def predict_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    class_index = np.argmax(prediction)
    confidence = round(np.max(prediction) * 100, 2)

    return classes[class_index], confidence

# =========================
# MAIN ROUTE
# =========================
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():

    prediction_data = []

    if request.method == 'POST' and 'image' in request.files:

        file = request.files['image']

        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = Image.open(filepath).convert('RGB')
        result, confidence = predict_image(img)

        # clean label
        clean_label = result.split("_", 1)[-1]
        clean_label = clean_label.replace("_", " ").capitalize()

        solution = solutions.get(clean_label, "Consult expert")

        # SAVE TO DATABASE ✅
        new_prediction = Prediction(
            user_id=current_user.id,
            label=clean_label,
            confidence=confidence,
            image=filename
        )

        db.session.add(new_prediction)
        db.session.commit()

        # SEND TO FRONTEND ✅
        prediction_data = [{
            "label": clean_label,
            "confidence": confidence,
            "image": filename,
            "solution": solution
        }]

    return render_template("index.html", predictions=prediction_data)

# =========================
# HISTORY
# =========================
@app.route('/history')
@login_required
def history():
    data = Prediction.query.filter_by(user_id=current_user.id).all()
    return render_template("history.html", history=data)

# =========================
# PDF DOWNLOAD
# =========================
@app.route('/download_pdf')
@login_required
def download_pdf():
    data = Prediction.query.filter_by(user_id=current_user.id).all()

    pdf_path = "static/history.pdf"

    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("PlantGuard - Prediction History", styles['Title']))
    content.append(Spacer(1, 20))

    for item in data:
        content.append(Paragraph(f"Disease: {item.label}", styles['Normal']))
        content.append(Paragraph(f"Confidence: {item.confidence}%", styles['Normal']))

        img_path = os.path.join("static/uploads", item.image)
        if os.path.exists(img_path):
            content.append(RLImage(img_path, width=200, height=150))

        content.append(Spacer(1, 20))

    doc.build(content)

    return redirect("/static/history.pdf")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(debug=True)