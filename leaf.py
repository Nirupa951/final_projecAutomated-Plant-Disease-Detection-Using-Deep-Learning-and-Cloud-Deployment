import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Plant Disease Detector", layout="centered")
st.title("🌿 Plant Disease Detection (Robust Inference)")

# === 38 classes (match your training EXACTLY; note lowercase 'healthy') ===
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# === Practical recommendations (optional) ===
RECO = {
    'Apple___Apple_scab': 'Spray Captan/Mancozeb; remove fallen leaves.',
    'Apple___Black_rot': 'Remove infected fruits; apply copper fungicide.',
    'Apple___Cedar_apple_rust': 'Apply Mancozeb; remove nearby junipers.',
    'Apple___healthy': 'Healthy 🌱 Maintain regular care.',
    'Blueberry___healthy': 'Healthy 🌱',
    'Cherry_(including_sour)___Powdery_mildew': 'Use sulfur; improve airflow.',
    'Cherry_(including_sour)___healthy': 'Healthy 🌱',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Apply Mancozeb; rotate crops.',
    'Corn_(maize)___Common_rust_': 'Use resistant hybrids; apply fungicide.',
    'Corn_(maize)___Northern_Leaf_Blight': 'Use Mancozeb; remove infected leaves.',
    'Corn_(maize)___healthy': 'Healthy 🌱',
    'Grape___Black_rot': 'Remove infected leaves; spray Captan.',
    'Grape___Esca_(Black_Measles)': 'Prune infected wood; improve irrigation.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Spray copper weekly.',
    'Grape___healthy': 'Healthy 🌱',
    'Orange___Haunglongbing_(Citrus_greening)': 'No cure; remove tree; control psyllids.',
    'Peach___Bacterial_spot': 'Use copper; avoid wet leaves.',
    'Peach___healthy': 'Healthy 🌱',
    'Pepper,_bell___Bacterial_spot': 'Use copper; avoid overhead watering.',
    'Pepper,_bell___healthy': 'Healthy 🌱',
    'Potato___Early_blight': 'Chlorothalonil; remove debris.',
    'Potato___Late_blight': 'Use Mancozeb or Metalaxyl; destroy infected leaves.',
    'Potato___healthy': 'Healthy 🌱',
    'Raspberry___healthy': 'Healthy 🌱',
    'Soybean___healthy': 'Healthy 🌱',
    'Squash___Powdery_mildew': 'Apply Neem oil.',
    'Strawberry___Leaf_scorch': 'Trim leaves; apply Captan.',
    'Strawberry___healthy': 'Healthy 🌱',
    'Tomato___Bacterial_spot': 'Use copper; avoid wet foliage.',
    'Tomato___Early_blight': 'Chlorothalonil; remove affected leaves.',
    'Tomato___Late_blight': 'Mancozeb/Metalaxyl; destroy infected leaves.',
    'Tomato___Leaf_Mold': 'Increase airflow; apply copper.',
    'Tomato___Septoria_leaf_spot': 'Remove lower leaves; apply fungicide.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Spray Neem oil.',
    'Tomato___Target_Spot': 'Apply Mancozeb.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Control whiteflies; yellow sticky traps; Neem.',
    'Tomato___Tomato_mosaic_virus': 'Remove infected leaves; sanitize hands.',
    'Tomato___healthy': 'Healthy 🌱'
}

# === Model loader ===
@st.cache_resource
def load_model():
    # Must be a FULL model saved via model.save(...)
    model = tf.keras.models.load_model("baseline_model.h5", compile=False)
    # build graph (ensures model.inputs is set)
    _ = model(tf.zeros((1, 128, 128, 3)))
    return model

model = load_model()

# === Detect preprocessing needed ===
def model_has_rescaling(m: tf.keras.Model) -> bool:
    for lyr in m.layers:
        if isinstance(lyr, tf.keras.layers.Rescaling):
            return True
    return False

def model_looks_like_resnet(m: tf.keras.Model) -> bool:
    # naive heuristic: any layer name contains "resnet"
    return any("resnet" in lyr.name.lower() for lyr in m.layers)

HAS_RESCALING = model_has_rescaling(model)
LOOKS_RESNET = model_looks_like_resnet(model)

if LOOKS_RESNET:
    from tensorflow.keras.applications.resnet50 import preprocess_input
else:
    resnet_preprocess = None

st.caption(f"Preprocessing auto-detect → RescalingLayer: {HAS_RESCALING} | ResNet-like: {LOOKS_RESNET}")

# === Inference helpers ===
def preprocess_pil(img_pil: Image.Image, size=(128, 128)) -> np.ndarray:
    img = img_pil.convert("RGB").resize(size)
    arr = np.array(img).astype("float32")
    if HAS_RESCALING:
        # Model already rescales 0–255 → 0–1 internally
        pass
    else:
        if LOOKS_RESNET and resnet_preprocess is not None:
            arr = resnet_preprocess(arr)  # will produce [-1,1] with channel transforms
        else:
            arr = arr / 255.0
    return np.expand_dims(arr, axis=0)

def softmax_if_needed(pred: np.ndarray) -> np.ndarray:
    # If last layer is logits (no softmax), apply a safe softmax
    if pred.ndim == 2 and (pred.max() > 1.0 or pred.min() < 0.0):
        pred = tf.nn.softmax(pred, axis=-1).numpy()
    return pred

# === UI ===
uploaded = st.file_uploader("📤 Upload a leaf image (128×128 expected, any size ok)", type=["jpg", "jpeg", "png"])

if uploaded:
    pil_img = Image.open(uploaded)
    st.image(pil_img, caption="Uploaded", use_container_width=True)

    x = preprocess_pil(pil_img, size=(128, 128))
    pred = model.predict(x)
    pred = softmax_if_needed(pred)

    # Top-1
    i = int(np.argmax(pred[0]))
    prob = float(pred[0][i])
    label = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class_{i}"

    st.subheader(f"🧪 Detected Disease: {label}")
    st.write(f"📊 Confidence: {prob*100:.2f}%")

    # Show recommendation only when confident
    CONF_THRESH = 0.50
    if prob >= CONF_THRESH:
        st.success(f"🌱 Recommendation: {RECO.get(label, 'Maintain regular care.')}")
    else:
        st.warning(
            "Model confidence is low. Avoid specific treatment. "
            "Try a clearer leaf photo (single leaf, good lighting, plain background)."
        )

else:
    st.info("⬆️ Upload a leaf image to get prediction.")