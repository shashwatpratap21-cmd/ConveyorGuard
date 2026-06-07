import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- Keras Version Mismatch Hacks (Tumhara original working code) ---
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

@tf.keras.utils.register_keras_serializable()
class SafeDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

# --- App Configuration (Layout ab WIDE kar diya hai) ---
st.set_page_config(
    page_title="ConveyorGuard", 
    page_icon="⛏️", 
    layout="wide"
)

# --- SIDEBAR: Economic Impact Calculator ---
st.sidebar.header("📉 Economic Impact Calculator")
st.sidebar.markdown("Estimate financial loss during downtime.")

capacity = st.sidebar.number_input("Conveyor Capacity (TPH)", min_value=100, max_value=5000, value=600, step=50)
coal_price = st.sidebar.number_input("Coal Price (₹/t)", min_value=1000, max_value=10000, value=2200, step=100)
downtime = st.sidebar.number_input("Predicted Downtime (h)", min_value=0.5, max_value=24.0, value=3.0, step=0.5)

loss_in_lakhs = (capacity * coal_price * downtime) / 100000

st.sidebar.markdown("---")
st.sidebar.subheader("Estimated Production Loss:")
st.sidebar.error(f"₹ {loss_in_lakhs:.2f} Lakh")
st.sidebar.markdown("---")
st.sidebar.info("System built for Sijua Colliery Operations.")

# --- MAIN DASHBOARD HEADER ---
st.title("⛏️ ConveyorGuard Dashboard")
st.subheader("AI-Powered Inspection & Safety Management System")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'conveyorguard_model.h5')
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

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {str(e)}")
    st.stop()

# --- TABS LAYOUT ---
tab1, tab2, tab3 = st.tabs(["🚨 AI Vision Inspection", "📝 Manual Override (Codes)", "🛠️ Maintenance Scheduler"])

# --- TAB 1: AI INSPECTION (Tumhara working AI logic) ---
with tab1:
    st.markdown("### Upload Conveyor Belt Image")
    uploaded_file = st.file_uploader("Drag and drop or click to upload", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        # UI Upgrade: Image left side, Results right side
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption='Live Belt Feed', use_container_width=True)
        
        with col2:
            with st.spinner('Analyzing belt condition...'):
                img = image.resize((224, 224))
                img
