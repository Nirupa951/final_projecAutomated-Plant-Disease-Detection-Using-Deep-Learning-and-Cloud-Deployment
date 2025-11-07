import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Plant Disease Detector", layout="centered")
st.title("🌿 Plant Disease Detection")


# ✅ 38 PlantVillage Classes
class_names = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___Healthy",
    "Blueberry___Healthy",
    "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___Healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___Healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___Healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___Healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___Healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___Healthy",
    "Raspberry___Healthy", "Soybean___Healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___Healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___Healthy"
]

# ✅ Recommendations Dictionary (Practical Actions)
recommendations = {
    "Apple___Apple_scab": "Spray Captan or Mancozeb. Remove fallen leaves.",
    "Apple___Black_rot": "Remove infected fruits and apply copper fungicide.",
    "Apple___Cedar_apple_rust": "Apply Mancozeb. Remove nearby junipers.",
    "Apple___Healthy": "Healthy 🌱 Maintain regular care.",
    "Blueberry___Healthy": "Healthy 🌱",
    "Cherry_(including_sour)___Powdery_mildew": "Use sulfur spray. Improve airflow.",
    "Cherry_(including_sour)___Healthy": "Healthy 🌱",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Apply Mancozeb and rotate crops.",
    "Corn_(maize)___Common_rust_": "Use resistant hybrids and apply fungicide.",
    "Corn_(maize)___Northern_Leaf_Blight": "Use Mancozeb and remove infected leaves.",
    "Corn_(maize)___Healthy": "Healthy 🌱",
    "Grape___Black_rot": "Remove infected leaves and spray Captan.",
    "Grape___Esca_(Black_Measles)": "Prune infected wood and improve irrigation.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Spray copper fungicide weekly.",
    "Grape___Healthy": "Healthy 🌱",
    "Orange___Haunglongbing_(Citrus_greening)": "No cure. Remove infected tree. Control psyllids.",
    "Peach___Bacterial_spot": "Use copper spray and avoid wet leaves.",
    "Peach___Healthy": "Healthy 🌱",
    "Pepper,_bell___Bacterial_spot": "Use copper fungicide. Avoid overhead watering.",
    "Pepper,_bell___Healthy": "Healthy 🌱",
    "Potato___Early_blight": "Spray Chlorothalonil and remove debris.",
    "Potato___Late_blight": "Use Metalaxyl fungicide. Remove infected plants.",
    "Potato___Healthy": "Healthy 🌱",
    "Raspberry___Healthy": "Healthy 🌱",
    "Soybean___Healthy": "Healthy 🌱",
    "Squash___Powdery_mildew": "Apply Neem oil.",
    "Strawberry___Leaf_scorch": "Trim damaged leaves and apply Captan.",
    "Strawberry___Healthy": "Healthy 🌱",
    "Tomato___Bacterial_spot": "Use copper fungicide. Avoid wet foliage.",
    "Tomato___Early_blight": "Apply Chlorothalonil. Remove affected leaves.",
    "Tomato___Late_blight": "Use Mancozeb or Metalaxyl. Destroy infected leaves.",
    "Tomato___Leaf_Mold": "Increase airflow & apply copper fungicide.",
    "Tomato___Septoria_leaf_spot": "Remove lower leaves. Apply fungicide.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Spray Neem oil.",
    "Tomato___Target_Spot": "Apply Mancozeb.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Use yellow sticky traps & Neem oil to control whiteflies.",
    "Tomato___Tomato_mosaic_virus": "Remove infected leaves. Wash hands before touching plants.",
    "Tomato___Healthy": "Healthy 🌱"
}


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("baseline_model.h5", compile=False)
    model(tf.zeros((1,128,128,3)))   # ✅ Fix: ensures model.input exists
    return model

model = load_model()


uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", use_container_width=True)

    img = cv2.resize(np.array(image), (128, 128))
    img_input = np.expand_dims(img / 255.0, axis=0)

    preds = model.predict(img_input)
    idx = np.argmax(preds[0])
    label = class_names[idx]
    confidence = np.max(preds[0]) * 100

    st.success(f"🧪 **Detected Disease:** {label}")
    st.info(f"📊 **Confidence:** {confidence:.2f}%")
    st.warning(f"🌱 **Recommendation:** {recommendations[label]}")

st.markdown("---")
st.caption("Developed by Nirupa 🌱 | AI-Powered Agriculture")