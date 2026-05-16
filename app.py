# -*- coding: utf-8 -*-
import os
import sys

os.system('pip install gradio tensorflow opencv-python-headless numpy Pillow')

import tensorflow as tf
import numpy as np
import gradio as gr
from PIL import Image

model_path = 'plant_expert_model.h5'

if os.path.exists(model_path):
    print("🚀 Loading pre-trained model...")
    model = tf.keras.models.load_model(model_path)
else:
    print("❌ Error: plant_expert_model.h5 not found! Please upload it to GitHub.")
    sys.exit(1)

class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 'Corn___Cercospora_leaf_spot',
    'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

disease_advice = {
    "healthy": "Your plant is healthy. Continue proper watering, sunlight, and regular monitoring.",
    "early_blight": "Remove infected leaves, avoid overwatering, and apply fungicide regularly.",
    "late_blight": "Isolate the plant immediately and improve airflow around the plant.",
    "blight": "Prune infected areas and avoid wetting leaves during watering.",
    "leaf_spot": "Remove spotted leaves and apply a suitable disease-control spray.",
    "septoria": "Avoid overhead watering and remove infected foliage immediately.",
    "target_spot": "Improve ventilation and use fungicide if the infection spreads.",
    "bacterial_spot": "Remove infected leaves and disinfect gardening tools after use.",
    "mold": "Reduce humidity and increase air circulation around the plant.",
    "leaf_mold": "Keep leaves dry and avoid overcrowding plants.",
    "powdery_mildew": "Place plant in sunlight and apply sulfur-based fungicide if necessary.",
    "downy_mildew": "Avoid excess moisture and water the soil instead of leaves.",
    "mildew": "Improve airflow and avoid keeping leaves wet for long periods.",
    "root_rot": "Reduce watering immediately and improve soil drainage.",
    "rot": "Check soil moisture and remove rotten plant parts.",
    "rust": "Remove infected leaves and avoid splashing water on foliage.",
    "virus": "Isolate the infected plant to prevent disease spread.",
    "yellow_leaf_curl_virus": "Control whiteflies and remove infected leaves quickly.",
    "mosaic_virus": "Disinfect tools and isolate infected plants immediately.",
    "bacterial": "Avoid touching wet plants and disinfect all gardening equipment.",
    "canker": "Prune infected branches carefully and improve plant health.",
    "aphid": "Use neem oil or insecticidal soap to control aphids safely.",
    "mite": "Spray water under leaves and use suitable mite treatment.",
    "spider_mites": "Increase humidity slightly and clean affected leaves.",
    "whitefly": "Use sticky traps and isolate heavily infected plants.",
    "thrips": "Apply insecticidal soap and remove heavily damaged leaves.",
    "mealybug": "Clean leaves using alcohol-dipped cotton and apply neem oil.",
    "scale": "Scrape insects gently and apply horticultural oil.",
    "caterpillar": "Remove caterpillars manually and inspect leaves regularly.",
    "nitrogen": "Use nitrogen-rich fertilizer to improve leaf growth.",
    "potassium": "Apply potassium fertilizer to strengthen plant resistance.",
    "calcium": "Add calcium supplements to prevent leaf and fruit damage.",
    "magnesium": "Use magnesium fertilizer or Epsom salt treatment.",
    "iron": "Apply iron-rich fertilizer to restore green leaf color.",
    "wilting": "Check watering schedule and inspect roots for damage.",
    "drought": "Increase watering frequency and add mulch to retain moisture.",
    "sunburn": "Move plant away from intense sunlight during hot hours.",
    "scorch": "Protect the plant from heat stress and dry winds.",
    "cold_damage": "Protect plants from low temperatures and frost exposure.",
    "heat_stress": "Provide shade and water during cooler times of the day.",
    "fungal": "Apply fungicide and reduce humidity around the plant.",
    "disease": "Monitor the plant daily and remove infected parts quickly.",
    "infection": "Keep infected plants separated from healthy plants."
}

def get_advice(label):
    label_lower = label.lower()
    for key in disease_advice:
        if key in label_lower:
            return disease_advice[key]
    return "No specific advice available for this disease."

def predict_disease(img, location):
    if img is None:
        return "<div style='text-align:center; padding:20px; color:red;'>Please upload an image.</div>"
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array, verbose=0)
    score = predictions[0]
    class_idx = np.argmax(score)
    label = class_names[class_idx]
    confidence = float(score[class_idx]) * 100
    clean_label = label.replace("___", " ")
    
    if "healthy" in label.lower():
        status = "🌱 HEALTHY"
        status_color = "green"
    else:
        status = "⚠️ DISEASE DETECTED"
        status_color = "red"
        
    advice = get_advice(label)
    
    result = f"""
    <div style="padding:20px; border-radius:15px; background-color:#f8f9fa; border:1px solid #dcdcdc; font-family:Arial, sans-serif;">
    <h2 style="text-align:center; color:#2e7d32; margin-bottom: 20px;">🌿 Plant Disease Detection Report</h2>
    <div style="background-color:#e9ecef; padding:15px; border-radius:10px; margin-bottom:15px;">
        <h3 style="color:#495057; margin-top:0;">🔬 Detected Disease:</h3>
        <p style="font-size:18px; font-weight:bold; color:#343a40;">{clean_label}</p>
    </div>
    <div style="background-color:#e9ecef; padding:15px; border-radius:10px; margin-bottom:15px;">
        <h3 style="color:#495057; margin-top:0;">📊 Confidence:</h3>
        <p style="font-size:18px; font-weight:bold; color:#343a40;">{confidence:.2f}%</p>
    </div>
    <div style="background-color:#e9ecef; padding:15px; border-radius:10px; margin-bottom:15px;">
        <h3 style="color:#495057; margin-top:0;">📌 Current Status:</h3>
        <p style="font-size:20px; font-weight:bold; color:{status_color};">{status}</p>
    </div>
    <div style="background-color:#e9ecef; padding:15px; border-radius:10px; margin-bottom:15px;">
        <h3 style="color:#495057; margin-top:0;">📍 Plant Location:</h3>
        <p style="font-size:18px; font-weight:bold; color:#343a40;">{location if location else 'Outdoor'}</p>
    </div>
    <div style="background-color:#e9ecef; padding:15px; border-radius:10px;">
        <h3 style="color:#495057; margin-top:0;">💡 Recommended Advice:</h3>
        <p style="font-size:17px; line-height:1.7; color:#343a40;">{advice}</p>
    </div>
    </div>
    """
    return result

theme = gr.themes.Soft(primary_hue="green", secondary_hue="emerald", neutral_hue="slate")

interface = gr.Interface(
    fn=predict_disease,
    inputs=[
        gr.Image(type="pil", label="Upload Leaf Image", height=250),
        gr.Dropdown(["Indoor", "Outdoor", "Greenhouse"], label="Plant Location", value="Outdoor")
    ],
    outputs=gr.HTML(label="Detection Result"),
    title="GHARS🌿",
    description="Welcome to GHARS: Growing smarter care for healthier plants 🌱",
    theme=theme,
    flagging_mode="never"
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=8080)
