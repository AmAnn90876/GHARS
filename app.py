# -*- coding: utf-8 -*-
import os
import sys
import zipfile

os.system('pip install gradio tensorflow opencv-python-headless numpy scikit-learn seaborn matplotlib')

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image
from tensorflow.keras.preprocessing import image
import gradio as gr

if not os.path.exists("archive.zip"):
    print("Error: Please upload archive.zip to your GitHub repository.")
    sys.exit(1)

with zipfile.ZipFile("archive.zip", 'r') as zip_ref:
    zip_ref.extractall("dataset")

print("تم فك الضغط بنجاح في مجلد dataset!")

data_dir = 'dataset/plantvillage dataset/color'

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(224, 224),
  batch_size=32)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(224, 224),
  batch_size=32)

class_names = train_ds.class_names
print(f"✅ Setup Complete: Found {len(class_names)} classes.")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Rescaling(1./255), 
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2), 
    layers.Dense(len(class_names), activation='softmax') 
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("🚀 Starting Professional Training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=7,
    callbacks=[early_stop]
)

model.save('plant_expert_model.h5')
print("✅ Training Finished and Model Saved!")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy', color='blue')
plt.plot(val_acc, label='Validation Accuracy', color='orange')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss', color='blue')
plt.plot(val_loss, label='Validation Loss', color='orange')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('performance_curves.png')
plt.close()

all_true_labels = []
all_predictions = []

for images_batch, labels_batch in val_ds:
    all_true_labels.extend(labels_batch.numpy())
    preds = model.predict(images_batch, verbose=0) 
    all_predictions.extend(np.argmax(preds, axis=1))

all_true_labels = np.array(all_true_labels)
all_predictions = np.array(all_predictions)

cm = confusion_matrix(all_true_labels, all_predictions)

plt.figure(figsize=(20, 18))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available ✅")
        print("Using:", gpus[0])
    except RuntimeError as e:
        print(e)
else:
    print("GPU is NOT available ❌")

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
    img_array = image.img_to_array(img)
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
        <p style="font-size:18px; font-weight:bold; color:#343a40;">{location if location else 'غير محدد'}</p>
    </div>
    <div style="background-color:#e9ecef; padding:15px; border-radius:10px;">
        <h3 style="color:#495057; margin-top:0;">💡 Recommended Advice:</h3>
        <p style="font-size:17px; line-height:1.7; color:#343a40;">{advice}</p>
    </div>
    </div>
    """
    return result

theme = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="emerald",
    neutral_hue="slate"
)

interface = gr.Interface(
    fn=predict_disease,
    inputs=[
        gr.Image(type="pil", label="Upload Leaf Image", height=250),
        gr.Dropdown(["Indoor", "Outdoor", "Greenhouse"], label="Plant Location", value="Outdoor")
    ],
    outputs=gr.HTML(label="Detection Result"),
    title="GHARS🌿",
    description="Upload an image of a plant to detect diseases and get advice.",
    theme=theme,
    flagging_mode="never"
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=8080)
