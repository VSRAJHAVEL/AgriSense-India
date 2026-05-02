"""
Flask Application - India Crop Recommendation System
Handles soil image classification and crop recommendation.
"""
import os
import json
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ─── Soil type to NPK/climate mapping (India-wide) ──────────────
SOIL_PROPERTIES = {
    "Alluvial soil": {
        "display_name": "Alluvial Soil",
        "description": "Most widespread and fertile soil in India, found across Indo-Gangetic plains, Brahmaputra valley, and coastal deltas. Rich in potash, phosphoric acid, and lime. Ideal for intensive agriculture.",
        "N_range": (60, 90), "P_range": (35, 55), "K_range": (35, 65),
        "ph_range": (6.0, 7.5), "temp_range": (22, 35),
        "humidity_range": (55, 85), "rainfall_range": (800, 1600),
        "color": "#C4A882", "regions": "Punjab, Haryana, UP, Bihar, West Bengal, Assam, Tamil Nadu, Andhra Pradesh, Coastal Gujarat"
    },
    "Black Soil": {
        "display_name": "Black Soil (Regur)",
        "description": "Also called Regur or Black Cotton Soil. Formed from Deccan Trap basalt. Rich in calcium, magnesium, potash, and iron. Excellent moisture retention. Covers the Deccan Plateau region.",
        "N_range": (40, 70), "P_range": (40, 60), "K_range": (40, 60),
        "ph_range": (6.5, 8.0), "temp_range": (24, 37),
        "humidity_range": (40, 70), "rainfall_range": (500, 1200),
        "color": "#3D3D3D", "regions": "Maharashtra, Gujarat, Madhya Pradesh, Karnataka, Andhra Pradesh, Tamil Nadu, Rajasthan"
    },
    "Clay soil": {
        "display_name": "Clay Soil",
        "description": "Dense, heavy soil with high water-holding capacity. Found in river deltas, wetlands, and low-lying areas across India. Excellent for paddy, sugarcane, and other water-intensive crops.",
        "N_range": (55, 85), "P_range": (35, 55), "K_range": (40, 70),
        "ph_range": (6.0, 7.5), "temp_range": (23, 35),
        "humidity_range": (60, 90), "rainfall_range": (900, 1800),
        "color": "#8B6914", "regions": "Sundarbans, Cauvery Delta, Krishna-Godavari Delta, Kerala Backwaters, Coastal Odisha"
    },
    "Red soil": {
        "display_name": "Red Soil",
        "description": "Formed from weathering of crystalline and metamorphic rocks. Rich in iron oxide giving the red color. Generally poor in nitrogen, phosphorus, and humus. Covers large parts of peninsular India.",
        "N_range": (20, 45), "P_range": (25, 50), "K_range": (20, 45),
        "ph_range": (5.5, 7.0), "temp_range": (24, 38),
        "humidity_range": (40, 70), "rainfall_range": (400, 1000),
        "color": "#C0392B", "regions": "Tamil Nadu, Karnataka, Andhra Pradesh, Odisha, Jharkhand, Chhattisgarh, Madhya Pradesh, Rajasthan"
    },
    "Loamy soil": {
        "display_name": "Loamy Soil",
        "description": "The ideal agricultural soil with balanced mix of sand, silt, and clay. Excellent drainage, moisture retention, and nutrient content. Most versatile soil type supporting a wide range of crops across India.",
        "N_range": (50, 80), "P_range": (30, 55), "K_range": (30, 60),
        "ph_range": (6.0, 7.5), "temp_range": (20, 35),
        "humidity_range": (50, 80), "rainfall_range": (600, 1400),
        "color": "#8B7355", "regions": "Punjab, Haryana, UP, Bihar, Madhya Pradesh, Rajasthan, Gujarat, Maharashtra"
    }
}

# ─── Load Models ────────────────────────────────────────────────
soil_model = None
soil_classes = None
crop_model_data = None
crop_details = None

def load_models():
    global soil_model, soil_classes, crop_model_data, crop_details
    
    # Load soil classifier
    soil_model_path = "models/soil_classifier.h5"
    soil_classes_path = "models/soil_classes.json"
    if os.path.exists(soil_model_path):
        import tensorflow as tf
        soil_model = tf.keras.models.load_model(soil_model_path)
        with open(soil_classes_path, "r") as f:
            soil_classes = json.load(f)
        print("[OK] Soil classifier loaded")
    else:
        print("[WARN] Soil classifier not found - using rule-based fallback")
    
    # Load crop recommender
    crop_model_path = "models/crop_recommender.pkl"
    if os.path.exists(crop_model_path):
        crop_model_data = joblib.load(crop_model_path)
        print("[OK] Crop recommender loaded")
    else:
        print("[WARN] Crop recommender not found - run train_crop_model.py first")
    
    # Load crop details
    details_path = "data/crop_details.json"
    if os.path.exists(details_path):
        with open(details_path, "r", encoding="utf-8") as f:
            crop_details = json.load(f)
        print(f"[OK] Crop details loaded ({len(crop_details)} crops)")
    else:
        print("[WARN] Crop details not found")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def classify_soil(image_path):
    """Classify soil type from image using CNN or color-based fallback."""
    img = Image.open(image_path).convert('RGB')
    
    if soil_model is not None and soil_classes is not None:
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = soil_model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        soil_type = soil_classes[str(predicted_idx)]
        confidence = float(predictions[predicted_idx]) * 100
        
        all_predictions = {}
        for idx, prob in enumerate(predictions):
            all_predictions[soil_classes[str(idx)]] = round(float(prob) * 100, 2)
        
        return soil_type, confidence, all_predictions
    else:
        # Color-based fallback analysis
        img_small = img.resize((100, 100))
        pixels = np.array(img_small).reshape(-1, 3)
        avg_r, avg_g, avg_b = pixels.mean(axis=0)
        
        brightness = (avg_r + avg_g + avg_b) / 3
        
        if brightness < 80:
            soil_type = "Black Soil"
            confidence = 70.0
        elif avg_r > avg_g * 1.3 and avg_r > avg_b * 1.3:
            soil_type = "Red soil"
            confidence = 65.0
        elif avg_r > 150 and avg_g > 120 and avg_b < 130:
            soil_type = "Alluvial soil"
            confidence = 60.0
        else:
            soil_type = "Clay soil"
            confidence = 55.0
        
        all_predictions = {st: 10.0 for st in SOIL_PROPERTIES}
        all_predictions[soil_type] = confidence
        remaining = 100 - confidence
        others = [s for s in SOIL_PROPERTIES if s != soil_type]
        for s in others:
            all_predictions[s] = round(remaining / len(others), 2)
        
        return soil_type, confidence, all_predictions

def recommend_crops(soil_type):
    """Recommend crops based on soil type using the trained model."""
    if soil_type not in SOIL_PROPERTIES:
        soil_type = list(SOIL_PROPERTIES.keys())[0]
    
    props = SOIL_PROPERTIES[soil_type]
    
    if crop_model_data is not None:
        model = crop_model_data["model"]
        le = crop_model_data["label_encoder"]
        
        # Generate multiple parameter samples and get probability-based recommendations
        samples = []
        for _ in range(50):
            sample = {
                "N": np.random.randint(*props["N_range"]),
                "P": np.random.randint(*props["P_range"]),
                "K": np.random.randint(*props["K_range"]),
                "temperature": np.random.uniform(*props["temp_range"]),
                "humidity": np.random.uniform(*props["humidity_range"]),
                "ph": np.random.uniform(*props["ph_range"]),
                "rainfall": np.random.uniform(*props["rainfall_range"]),
            }
            samples.append(list(sample.values()))
        
        X = np.array(samples)
        predictions = model.predict(X)
        
        # Count frequency of each crop prediction
        crop_counts = {}
        for pred in predictions:
            crop_name = le.inverse_transform([pred])[0]
            crop_counts[crop_name] = crop_counts.get(crop_name, 0) + 1
        
        # Sort by frequency and compute suitability percentage
        total = len(predictions)
        recommended = []
        for crop_name, count in sorted(crop_counts.items(), key=lambda x: x[1], reverse=True):
            suitability = round((count / total) * 100, 1)
            details = crop_details.get(crop_name, {}) if crop_details else {}
            recommended.append({
                "name": crop_name,
                "suitability": suitability,
                "details": details
            })
        
        return recommended[:8]  # Top 8 crops
    else:
        # Fallback: recommend based on soil type matching in crop_details
        recommended = []
        if crop_details:
            soil_name_map = {
                "Alluvial soil": "Alluvial",
                "Black Soil": "Black",
                "Clay soil": "Clay",
                "Red soil": "Red"
            }
            soil_key = soil_name_map.get(soil_type, "Red")
            
            for crop_id, details in crop_details.items():
                if soil_key in details.get("soil_types", []):
                    recommended.append({
                        "name": crop_id,
                        "suitability": 75.0,
                        "details": details
                    })
        
        return recommended[:8]

# ─── Routes ─────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "soil_image" not in request.files:
        return redirect(url_for("index"))
    
    file = request.files["soil_image"]
    if file.filename == "" or not allowed_file(file.filename):
        return redirect(url_for("index"))
    
    # Save uploaded file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"soil_{timestamp}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Classify soil
    soil_type, confidence, all_predictions = classify_soil(filepath)
    soil_props = SOIL_PROPERTIES.get(soil_type, {})
    
    # Get crop recommendations
    recommendations = recommend_crops(soil_type)
    
    return render_template("result.html",
        soil_type=soil_type,
        soil_display=soil_props.get("display_name", soil_type),
        soil_description=soil_props.get("description", ""),
        soil_color=soil_props.get("color", "#666"),
        soil_regions=soil_props.get("regions", ""),
        confidence=round(confidence, 1),
        all_predictions=all_predictions,
        soil_props=soil_props,
        recommendations=recommendations,
        image_path=f"uploads/{filename}",
        timestamp=datetime.now().strftime("%d %B %Y, %I:%M %p")
    )

@app.route("/api/crop/<name>")
def crop_api(name):
    if crop_details and name in crop_details:
        return jsonify(crop_details[name])
    return jsonify({"error": "Crop not found"}), 404

@app.route("/api/crops")
def all_crops_api():
    if crop_details:
        return jsonify(crop_details)
    return jsonify({"error": "No crop data"}), 404

# ─── Main ───────────────────────────────────────────────────────
# Load models immediately so they are available when running via Gunicorn
load_models()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

