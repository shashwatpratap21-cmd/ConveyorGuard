import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- Keras Version Mismatch Hacks ---
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

# --- App Configuration ---
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

# --- TAB 1: AI INSPECTION ---
with tab1:
    st.markdown("### Upload Conveyor Belt Image")
    
    # --- NEW: Safety Precautions directly below the description ---
    st.info("""
    **👷‍♂️ Pre-Inspection Safety Checklist (DGMS Guidelines):**
    * Ensure you are standing in a designated safe walkway.
    * Do not bypass physical safety guards to take photos.
    * Maintain a minimum 1.5m clearance from moving idlers.
    """)
    
    uploaded_file = st.file_uploader("Drag and drop or click to upload", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption='Live Belt Feed', use_container_width=True)
        
        with col2:
            with st.spinner('Analyzing belt condition...'):
                img = image.resize((224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)
                
                prediction = model.predict(img_array)[0][0]
                
                st.subheader("Inspection Verdict:")
                
                if prediction < 0.5:
                    confidence = (1 - prediction) * 100
                    
                    if confidence >= 85.0:
                        st.error(f"🔴 CRITICAL RUPTURE ({confidence:.1f}% Confidence)")
                        # --- NEW: Time to Failure Metric ---
                        st.metric("⏳ Estimated Time to Failure", "IMMEDIATE (0 Hours)")
                        st.warning("**⚙️ Actions:** Trip pull-cord immediately. Isolate drive power.")
                        st.error("**👷‍♂️ Protocol:** Evacuate 50m radius. Execute LOTO. Activate sprinklers.")
                    elif confidence >= 70.0:
                        st.warning(f"🟠 MODERATE DAMAGE ({confidence:.1f}% Confidence)")
                        # --- NEW: Time to Failure Metric ---
                        st.metric("⏳ Estimated Time to Failure", "48 - 72 Hours")
                        st.info("**⚙️ Actions:** Reduce speed by 50%. Schedule shift-end repair.")
                        st.warning("**👷‍♂️ Protocol:** Clear walkway beneath flagged section.")
                    else:
                        st.info(f"🟡 MINOR WEAR ({confidence:.1f}% Confidence)")
                        # --- NEW: Time to Failure Metric ---
                        st.metric("⏳ Estimated Time to Failure", "2 - 4 Weeks")
                        st.info("**⚙️ Actions:** Continue operations. Log for weekend maintenance.")
                        
                else:
                    confidence = prediction * 100
                    st.success(f"✅ NORMAL ({confidence:.1f}% Confidence)")
                    # --- NEW: Time to Failure Metric ---
                    st.metric("⏳ Estimated Time to Failure", "> 6 Months")
                    st.info("Belt condition healthy. Continue production.")
# --- TAB 2: MANUAL OVERRIDE ---
with tab2:
    st.markdown("### Emergency Manual Reporting")
    st.write("Use this if the camera is covered in dust or network is too slow for image uploads.")
    
    incident_code = st.text_input("Enter 4-Letter Emergency Code (e.g., TEAR, SPIL, FIRE):").upper()
    
    if incident_code:
        if incident_code == "TEAR":
            st.error("🚨 CODE TEAR LOGGED: Severe belt rupture reported manually.")
            st.warning("Action: Stop belt. Dispatch vulcanizing crew immediately.")
        elif incident_code == "SPIL":
            st.warning("⚠️ CODE SPIL LOGGED: Coal spillage blocking idlers.")
            st.info("Action: Dispatch cleaning crew to avoid friction fires.")
        elif incident_code == "FIRE":
            st.error("🔥 CODE FIRE LOGGED: Friction smoke detected.")
            st.error("CRITICAL: Evacuate district. Turn on main suppression systems.")
        else:
            st.info(f"Log received: {incident_code}. Control room notified for verification.")

# --- TAB 3: MAINTENANCE SCHEDULER ---
with tab3:
    st.markdown("### ⚙️ Time-Based Preventive Maintenance")
    
    col3, col4 = st.columns(2)
    with col3:
        belt_type = st.selectbox("Select Conveyor Belt Type", ["PVC Fire-Resistant (Underground)", "Steel Cord", "Nylon/Fabric Belt"])
    with col4:
        machine_age = st.number_input("Operating Time (Months)", min_value=1, max_value=120, value=6)
    
    st.markdown("---")
    st.subheader(f"📋 Maintenance Schedule for {machine_age} Months Old {belt_type}")
    
    if machine_age <= 6:
        st.info("**Routine Checks (0-6 Months):**\n* Visual inspection of belt tracking.\n* Check for unusual noise from idlers.")
    elif 6 < machine_age <= 24:
        st.warning("**Mid-Term Maintenance (6-24 Months):**\n* Lubricate all tail and drive pulleys.\n* Ultrasonic thickness test on belt cover.")
    else:
        st.error("**Critical Overhaul (>24 Months):**\n* Full structural audit.\n* Replace seized idlers.\n* Test all emergency pull-cords.")
