import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Setup the Page
st.set_page_config(page_title="ConveyorGuard", page_icon="⛏️", layout="centered")

st.title("⛏️ ConveyorGuard")
st.subheader("AI Belt Inspection System - Sijua Colliery Application")
st.write("Upload a photo of the conveyor belt for instant AI analysis.")

# 2. Load the Model 
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('conveyorguard_model.h5')

try:
    model = load_model()
except Exception as e:
    st.error("Model not found. Please ensure 'conveyorguard_model.h5' is uploaded.")

# 3. Create the Upload Button
uploaded_file = st.file_uploader("Drag and drop or click to upload", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Belt Photo', use_container_width=True)
    
    with st.spinner('Analyzing belt condition...'):
        # 4. Preprocess the image
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) 
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # 5. Make the Prediction
        prediction = model.predict(img_array)[0][0]
        
        st.markdown("---")
        st.subheader("Inspection Result:")
        
        # 6. Display the Output (0 = Anomaly, 1 = Normal)
        if prediction < 0.5:
            confidence = (1 - prediction) * 100
            st.error(f"🚨 **ANOMALY DETECTED**")
            st.write(f"**Confidence:** {confidence:.1f}%")
            st.write("⚠️ **Action:** Stop belt immediately. Notify shift engineer.")
        else:
            confidence = prediction * 100
            st.success(f"✅ **NORMAL**")
            st.write(f"**Confidence:** {confidence:.1f}%")
            st.write("Belt condition healthy. Production can continue.")