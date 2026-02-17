import streamlit as st
import pickle
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf  # for Keras model

# Page config
st.set_page_config(page_title="Jewelry Identifier", layout="wide")

st.markdown("""
# 🔮 Jewelry Identifier
**by 23MIA1120**  
Upload jewelry image → Auto-detect type!
""", unsafe_allow_html=True)

# Load model (safe version)
@st.cache_resource
def load_model():
    try:
        with open("my_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ my_model.pkl missing! Upload to repo root.")
        return None
    except Exception as e:
        st.error(f"❌ Load error: {str(e)[:100]}...")
        return None

model = load_model()
if model is None:
    st.stop()

# Sidebar for debug info
with st.sidebar:
    st.header("🛠️ Debug")
    if model:
        st.success("✅ Model loaded!")
        st.write("**Input shape:**", getattr(model, 'input_shape', 'N/A'))

# Main uploader
uploaded_file = st.file_uploader("📁 Upload jewelry image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display original
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded", use_column_width=True)
    
    # Process
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📊 Analysis")
        
        # OpenCV processing
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            aspect_ratio = img_array.shape[0] / img_array.shape[1]
            
            # **KEY: 5 MOST COMMON FEATURES for tabular models - ADJUST HERE**
            features = np.array([[area, perimeter, circularity, aspect_ratio, area/perimeter]])
            
            st.write("**Features shape:**", features.shape)
            st.write("**Features:**", features[0].round(3))
            
            try:
                # Predict (works for sklearn/Keras tabular)
                prediction = model.predict(features)[0]
                st.success(f"🎉 **Predicted: {prediction}**")
            except ValueError as e:
                st.error(f"❌ Shape error: {str(e)[:100]}")
                st.info("🔧 Fix: Match features to training data (check model.input_shape)")
        else:
            st.warning("No contours found.")
    
    with col2:
        st.header("👁️ Visual")
        if 'largest' in locals():
            img_contours = cv2.drawContours(img_array.copy(), [largest], -1, (0, 255, 0), 3)
            st.image(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB), caption="Contours", use_column_width=True)

# Footer
st.markdown("---")
st.markdown("*Built with ❤️ in Colab + Streamlit*")
