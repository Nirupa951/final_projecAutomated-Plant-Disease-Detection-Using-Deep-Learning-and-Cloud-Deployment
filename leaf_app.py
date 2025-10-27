import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🌱 Simple Leaf Disease Classifier")

# --- Load model ---
model = tf.keras.models.load_model("baseline_model.h5")

# --- Define 38 placeholder class names ---
class_names = [f"Class {i}" for i in range(38)]

# --- Upload image ---
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
if uploaded_file is None:
    st.info("Please upload an image.")
    st.stop()

# --- Preprocess image ---
img = Image.open(uploaded_file).convert("RGB").resize((128, 128))
img_array = np.array(img).astype("float32") / 255.0
img_batch = np.expand_dims(img_array, axis=0)

# --- Predict ---
pred = model.predict(img_batch)
cls_idx = np.argmax(pred[0])
confidence = np.max(pred[0]) * 100

# --- Safe label selection ---
label = class_names[cls_idx] if cls_idx < len(class_names) else f"Class {cls_idx}"

# --- Display ---
st.image(img, caption="Uploaded Leaf", use_container_width=True)
st.subheader(f"Prediction: {label}")
st.write(f"Confidence: {confidence:.2f}%")
st.success("Prediction complete!")