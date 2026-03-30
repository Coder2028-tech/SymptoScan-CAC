import os
import uuid
import shutil
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from google import genai
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
import secrets
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)



mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Utility functions (color, brightness, eye openness) ...
def get_color_mean(image, landmarks, indices):
    h, w, _ = image.shape
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    colors = [image[y, x] for x, y in points if 0 <= x < w and 0 <= y < h]
    return np.mean(colors, axis=0) if colors else np.array([0, 0, 0])

def get_brightness(image, landmarks, indices):
    h, w, _ = image.shape
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    brightness = [hsv[y, x][2] for x, y in points if 0 <= x < w and 0 <= y < h]
    return np.mean(brightness) if brightness else 0

def eye_openness(landmarks, top_idx, bottom_idx):
    return abs(landmarks[top_idx].y - landmarks[bottom_idx].y)


def detect_symptoms(filepath):
    image = cv2.imread(filepath)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return [], ""

    landmarks = results.multi_face_landmarks[0].landmark
    symptoms = []
    treatments = []

    # Redness in eyes
    eye_color = get_color_mean(image, landmarks, [33, 133])
    if eye_color[2] > 160:
        symptoms.append("Redness in eyes")
        treatments.append ("Use preservative-free artificial eye drops")

    # Dry lips
    lip_color = get_color_mean(image, landmarks, list(range(61, 88)))
    if lip_color[2] < 130 and lip_color[0] > 90:
        symptoms.append("Dry lips")
        treatments.append ("Use a fragrance-free lip balm containing ingredients like beeswax, petroleum jelly, or shea butter")

    # Swelling and fatigue
    eye_height = eye_openness(landmarks, 159, 145)
    if eye_height < 0.015:
        symptoms.append("Swelling")
        treatments.append ("Apply a cold compress and elevate the affected area")
    if eye_height < 0.018:
        symptoms.append("Fatigue")
        treatments.append ("Sleep well, limit salt intake")

    # Dry skin
    forehead_color = get_color_mean(image, landmarks, [10, 338, 297])
    saturation = cv2.cvtColor(np.uint8([[forehead_color]]), cv2.COLOR_BGR2HSV)[0][0][1]
    if saturation < 40:
        symptoms.append("Dry skin")
        treatments.append ("Apply a thick, fragrance-free moisturizer after taking a bath")

    # Runny nose
    brightness = get_brightness(image, landmarks, [195, 5, 4])
    if brightness > 180:
        symptoms.append("Runny Nose")
        treatments.append ("Use saline nasal spray")

    # Redness
    cheek_color = get_color_mean(image, landmarks, [205, 425])
    if cheek_color[2] > 160 and cheek_color[0] < 130:
        symptoms.append("Redness")
        treatments.append ("Use mild, hypoallergenic moisturizer if it's skin-related")

    # Create static folder if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')

    unique_filename = f"result_{uuid.uuid4().hex}.jpg"
    result_path = os.path.join("static", unique_filename)

    print(f"Saving result image to: {result_path}")
    success = cv2.imwrite(result_path, image)
    if not success:
        print(f"Error: Failed to save result image at {result_path}")
        return symptoms, None

    return symptoms, unique_filename, treatments

# Flask app setup
app = Flask(__name__, static_url_path='/static', static_folder='static')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev_secret_key_123") or secrets.token_hex(32)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///symptom_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    journals = db.relationship('JournalEntry', backref='user', lazy=True)

class JournalEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symptoms = db.Column(db.Text, nullable=False)  # store as JSON string
    discomfort = db.Column(db.Integer, nullable=True)  # <-- NEW FIELD
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)



@app.route('/')
def index():
    print("Index hit")
    return render_template('index.html')


@app.route('/analyze', methods=['POST', 'GET'])
def analyze():
    print("Analyze hit")
    if request.method == 'GET':
        return "Please use the form to upload an image.", 405

    file = request.files.get('image')
    captured_data = request.form.get('capturedImage')

    # Ensure at least one source is provided
    if (not file or file.filename.strip() == '') and not captured_data:
        return "No file uploaded", 400

    if file and file.filename.strip() != '':
        # Save uploaded file
        original_filename = f"original_{uuid.uuid4().hex}_{file.filename}"
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        file.save(original_path)
    else:
        # Handle captured image (base64 → file)
        import base64
        header, encoded = captured_data.split(",", 1)
        data = base64.b64decode(encoded)
        original_filename = f"captured_{uuid.uuid4().hex}.png"
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        with open(original_path, "wb") as f:
            f.write(data)

    # Run detection
    symptoms, result_filename, treatments = detect_symptoms(original_path)
    session["symptoms"] = symptoms

    if not result_filename:
        return "Error processing image.", 500

    return render_template("results.html", symptoms=symptoms, treatments=treatments, file=result_filename)




@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json.get("message", "")
    detected_symptoms = session.get("symptoms", [])

    # Format the symptoms into readable text
    symptoms_text = ", ".join(detected_symptoms) if detected_symptoms else "no symptoms detected"

    prompt = f"""
    You are Iris, a helpful medical insight chatbot.
    The user has reported or shown the following symptoms: {symptoms_text}.
    Your role is to provide educational information and possible explanations
    about these symptoms based on the user's questions.
    Do not diagnose or provide medical advice.
    Keep answers factual, concise, and between 1–4 sentences.

    Question: {user_msg}
    """

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )

    reply = response.text.strip()
    return jsonify({"reply": reply})



@app.route('/journal_page')
def journal_page():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('journal.html')

@app.route('/journal')
def journal():
    if 'user_id' not in session:
        return jsonify([])
    user_id = session['user_id']
    import json
    entries = JournalEntry.query.filter_by(user_id=user_id).order_by(JournalEntry.timestamp.desc()).all()
    result = []
    for entry in entries:
        result.append({
            "id": entry.id,
            "symptoms": json.loads(entry.symptoms),
            "discomfort": entry.discomfort,   # <-- include it
            "timestamp": entry.timestamp.isoformat()
        })

    return jsonify(result)

@app.route('/journal/add', methods=['POST'])
def add_journal():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_id = session['user_id']
    data = request.get_json()
    symptoms = data.get('symptoms')
    discomfort = data.get('discomfort')  # <-- get from request

    # Allow string or list
    if isinstance(symptoms, str):
        symptoms = [symptoms]
    if not symptoms or not isinstance(symptoms, list):
        return jsonify({"error": "Invalid symptoms"}), 400

    import json
    new_entry = JournalEntry(
        symptoms=json.dumps(symptoms),
        discomfort=discomfort,  # <-- save it
        user_id=user_id
    )
    db.session.add(new_entry)
    db.session.commit()

    return jsonify({"success": True, "id": new_entry.id})


@app.route('/journal', methods=['GET'])
def get_journal():
    if 'user_id' not in session:
        return jsonify([])

    user_id = session['user_id']
    import json
    entries = JournalEntry.query.filter_by(user_id=user_id).order_by(JournalEntry.timestamp.desc()).all()
    result = []
    for entry in entries:
        result.append({
            "id": entry.id,
            "symptoms": json.loads(entry.symptoms),
            "discomfort": entry.discomfort,   # <-- include it
            "timestamp": entry.timestamp.isoformat()
        })

    return jsonify(result)

@app.route('/journal/delete/<int:entry_id>', methods=['DELETE'])
def delete_journal_entry(entry_id):
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    entry = JournalEntry.query.get(entry_id)
    if not entry or entry.user_id != session['user_id']:
        return jsonify({"error": "Entry not found or access denied"}), 404
    db.session.delete(entry)
    db.session.commit()
    return jsonify({"success": True})


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session['username'] = username
            session['user_id'] = user.id
            flash("Logged in successfully!", "success")
            return redirect(url_for('journal_page'))
        else:
            flash("Invalid username or password.", "error")



    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    session.pop('username', None)
    flash("Logged out.", "info")
    return redirect(url_for('index'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        confirm = request.form.get('confirm_password', '').strip()

        if not username or not password:
            flash("Username and password are required.", "error")
            return render_template('signup.html')

        if password != confirm:
            flash("Passwords do not match.", "error")
            return render_template('signup.html')

        if User.query.filter_by(username=username).first():
            flash("Username already exists.", "error")
            return render_template('signup.html')

        hashed_pw = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password_hash=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        flash("Account created! Please log in.", "success")
        return redirect(url_for('login'))
    return render_template('signup.html')


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs('static', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT",5000)))
    print("Flask app created")
