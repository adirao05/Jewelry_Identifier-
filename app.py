import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Jewelry Identifier 23MIA1120", layout="wide")

st.markdown("""
<div style="text-align:center;background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);
            padding:2rem;border-radius:16px;color:white">
  <h1>Jewelry Identifier</h1>
  <p><em>Upload an image → model predicts the jewelry type (diamond ring or ruby pendant or sapphire earrings) </em></p>
</div>
""", unsafe_allow_html=True)

# MUST match training order (verify in Colab!)
CLASS_NAMES = ["diamond_ring", "ruby_pendant", "sapphire_earrings"]

MODEL_PATH = "jewelry_model.h5"  # your .h5 file name in GitHub repo root

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Model load error: {str(e)[:160]}...")
        return None

model = load_model()
if model is None:
    st.stop()

# Model expects (None, 64, 64, 3) as you showed
H, W, C = model.input_shape[1], model.input_shape[2], model.input_shape[3]

uploaded_file = st.file_uploader("📁 Upload jewelry image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Upload an image to start prediction.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="Uploaded image", use_column_width=True)

# Preprocess: (1, H, W, 3)
img = np.array(image)                      # RGB
img = cv2.resize(img, (W, H))
img = img.astype(np.float32) / 255.0
x = np.expand_dims(img, axis=0)

st.write("Input shape given to model:", x.shape)

try:
    probs = model.predict(x, verbose=0)[0]          # shape (num_classes,)
    pred_idx = int(np.argmax(probs))
    conf = float(np.max(probs))

    # Safe mapping
    pred_name = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"class_{pred_idx}"

    st.success(f"Prediction: {pred_name}")
    st.write(f"Confidence: {conf:.2%}")

    with st.expander("Show probabilities"):
        for i, p in enumerate(probs):
            label = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
            st.write(f"{label}: {float(p):.4f}")

except Exception as e:
    st.error(f"❌ Prediction error: {e}")
    st.code(f"model.input_shape = {model.input_shape}\nprovided x.shape = {x.shape}")




