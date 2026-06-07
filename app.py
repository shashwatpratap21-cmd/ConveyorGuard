import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- Keras Version Mismatch Hacks ---

# 1. INTERCEPTOR FIX: Hide the "127.5" division from strict rules
@tf.keras.utils.register_keras_serializable()
class TrueDivide(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def __call__(self, inputs, *args, **kwargs):
        if len(args) > 0:
            kwargs['y'] = args[0]
            args = tuple(args[1:])
        return super().__call__(inputs, *args, **kwargs)

    def call(self, inputs, y=127.5):
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            return inputs[0] / inputs[1]
        return inputs / y

# 2. INTERCEPTOR FIX: Hide the "1.0" subtraction from strict rules
@tf.keras.utils.register_keras_serializable()
class Subtract(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def __call__(self, inputs, *args, **kwargs):
        if len(args) > 0:
            kwargs['y'] = args[0]
            args = tuple(args[1:])
        return super().__call__(inputs, *args, **kwargs)

    def call(self, inputs, y=1.0):
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            return inputs[0] - inputs[1]
        return inputs - y

# 3. Fix for the Colab Dense layer bug
@tf.keras.utils.register_keras_serializable()
class SafeDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

# --- Main App Code ---

st.set_page_config(
    page_title="ConveyorGuard", 
    page_icon="⛏️", 
    layout="centered"
)

st.title("⛏️ ConveyorGuard")
st.subheader("AI Belt Inspection System - Sijua Colliery Application")
st.write("Upload a photo of the conveyor belt for instant AI analysis.")

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'conveyorguard_model.h5')
    
    # Load the model and apply our hacks
    model = tf.keras.models.load_model(
        model_path, 
        compile=False,
        safe_mode=False,
        custom_objects={
            'TrueDivide': TrueDivide,
            'Subtract': Subtract,
            'Dense': SafeDense
        }
    )
    return model

model = None
try:
    model = load_model()
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"Model loading failed: {str(e)}")
    st.stop()

uploaded_file = st.file_uploader(
    "Drag and drop or click to upload", 
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Belt Photo', use_container_width=True)
    
    with st.spinner('Analyzing belt condition...'):
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        prediction = model.predict(img_array)[0][0]
        
        st.markdown("---")
        st.subheader("Inspection Result:")
        
        if prediction < 0.5:
            confidence = (1 - prediction) * 100
            st.error(f"🚨 ANOMALY DETECTED")
            st.metric("Confidence", f"{confidence:.1f}%")
            st.warning("""
            **Immediate Actions Required:**
            1. Stop belt immediately
            2. Notify shift engineer
            3. Physically inspect flagged section
            4. Do not restart without clearance
            """)
        else:
            confidence = prediction * 100
            st.success(f"✅ NORMAL")
            st.metric("Confidence", f"{confidence:.1f}%")
            st.info("Belt condition healthy. Continue production.")
