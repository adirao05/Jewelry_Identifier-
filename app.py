import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# Page config
st.set_page_config(page_title="Jewelry Identifier", layout="wide")

st.markdown("""
<div style='text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; color: white;'>
    <h1>🔮 Jewelry Identifier</h1>
    <p><em>by 23MIA1120</em></p>
</div>
<br>
Upload jewelry image → AI identifies type!
""", unsafe_allow_html=True)

# Load Keras model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("jewelry_model.h5")  # or "my_model.pkl" if fixed
        return model
    except Exception as e:
        st.error(f"❌ Model error: {str(e)[:100]}...")
        return None

model = load_model()
if model is None:
    st.stop()

# Debug sidebar
with st.sidebar:
    st.header("🛠️ Model Info")
    st.success("✅ Model loaded!")
    st.json({"input_shape": str(model.input_shape)})

# File uploader
uploaded_file = st.file_uploader("📁 Upload image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original", use_column_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📊 Processing")
        
        # Image to array
        img_array = np.array(image)
        
        # CNN preprocessing (standard Keras: channels_last)
        target_size = (224, 224)  # UPDATE from model.input_shape[1:3]
        img_resized = cv2.resize(img_array, target_size) / 255.0
        features = np.expand_dims(img_resized, axis=0).astype(np.float32)  # (1, 224, 224, 3)
        
        st.write("**Input shape:**", features.shape)
        
        # Predict
        with st.spinner("Predicting..."):
            prediction = model.predict(features, verbose=0)
            pred_class = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
        st.success(f"🎉 **Jewelry Type: {pred_class}**")
        st.info(f"Confidence: {confidence:.1%}")
        st.write("**Raw probs:**", {i: f"{p:.1%}" for i, p in enumerate(prediction[0])})
    
    with col2:
        st.header("👁️ Preview")
        # Contours for fun
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            img_contours = cv2.drawContours(img_array.copy(), contours[:3], -1, (0, 255, 0), 3)
            st.image(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB), caption="Contours")
    
    # Download processed
    processed_img = (features[0] * 255).astype(np.uint8)
    st.download_button("💾 Download processed", 
                      cv2.imencode('.png', processed_img)[1].tobytes(), 
                      "processed.png")

st.markdown("---")
st.markdown("*Powered by Keras + Streamlit*")

