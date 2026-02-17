import streamlit as st
import pickle
import cv2
import numpy as np
from PIL import Image
import io  

import tensorflow as tf  # only if needed for predict

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("my_model.h5")
    except Exception as e:
        st.error(f"Load error: {str(e)}")
        return None

model = load_model()
if model is None:
    st.stop()

# Later: prediction = model.predict(features)


model = load_model()

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        features = np.array([[area, perimeter, circularity]]).reshape(1, -1)
        
        prediction = model.predict(features)[0] 
        st.success(f"Detected object: {prediction}")
        
        img_with_contours = cv2.drawContours(img_array.copy(), [largest_contour], -1, (0, 255, 0), 3)
        st.image(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB), caption="Contours")
    else:
        st.warning("No contours detected. Try a clearer image.")


