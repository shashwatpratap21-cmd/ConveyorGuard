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

# Financial Math
hourly_loss_lakhs = (capacity * coal_price) / 100000
total_loss_lakhs = hourly_loss_lakhs * downtime

st.sidebar.markdown("---")
st.sidebar.subheader("Estimated Production Loss:")
st.sidebar.error(f"🚨 ₹ {total_loss_lakhs:.2f} Lakh")

# Suggestion 1: Sijua Colliery Context
st.sidebar.caption("Based on Sijua Colliery average capacity of 600 TPH at ₹2,200/tonne coal price.")
st.sidebar.markdown("---")

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
        custom_objects={'TrueDivide': TrueDivide, 'Subtract': Subtract, 'Dense': SafeDense}
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
    
    # Suggestion 3: DGMS Yellow for Safety Checklist (Now Minimizable!)
    with st.expander("🟡 View Pre-Inspection Safety Checklist (DGMS Guidelines)", expanded=False):
        st.warning("""
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
                        # Suggestion 3: DGMS Red for Critical Alerts
                        st.error(f"🔴 CRITICAL RUPTURE ({confidence:.1f}% Confidence)")
                        
                        # Suggestion 2: Connect AI to Hourly Cost
                        st.error(f"💸 **Estimated cost if ignored:** ₹ {hourly_loss_lakhs:.2f} Lakh per hour of continued operation.")
                        
                        st.metric("⏳ Estimated Time to Failure", "IMMEDIATE (0 Hours)")
                        st.error("""
                        **🔴 Immediate Actions Required:**
                        * Trip pull-cord immediately. 
                        * Isolate drive power.
                        * Evacuate 50m radius. Execute LOTO. 
                        * Activate sprinklers.
                        """)
                    elif confidence >= 70.0:
                        st.warning(f"🟠 MODERATE DAMAGE ({confidence:.1f}% Confidence)")
                        st.metric("⏳ Estimated Time to Failure", "48 - 72 Hours")
                        st.warning("""
                        **🟠 Required Actions:**
                        * Reduce speed by 50%. 
                        * Clear walkway beneath flagged section.
                        * Schedule shift-end repair.
                        """)
                    else:
                        st.info(f"🟡 MINOR WEAR ({confidence:.1f}% Confidence)")
                        st.metric("⏳ Estimated Time to Failure", "2 - 4 Weeks")
                        
                        # Suggestion 3: DGMS Blue for Maintenance Advisory
                        st.info("""
                        **🔵 Maintenance Advisory:**
                        * Continue operations. 
                        * Log anomaly for weekend maintenance visual check.
                        """)
                        
                else:
                    confidence = prediction * 100
                    st.success(f"✅ NORMAL ({confidence:.1f}% Confidence)")
                    st.metric("⏳ Estimated Time to Failure", "> 6 Months")
                    st.success("Belt condition healthy. Continue production.")

# --- TAB 2: MANUAL OVERRIDE ---
with tab2:
    st.markdown("### 🎙️ Emergency Manual Reporting")
    st.write("Use your device's **Microphone (Dictation)**. Regional languages are supported.")
    
    # Language Toggle
    lang = st.radio("Response Language / उत्तर की भाषा:", ["English", "हिंदी"], horizontal=True)
    st.info("🇮🇳 Supported inputs: English, Hindi, or Hinglish (e.g., 'belt fat gaya', 'aag lag gayi', 'paani aa raha hai')")
    
    # 1. Initialize the Memory (Session State)
    if 'saved_report' not in st.session_state:
        st.session_state.saved_report = ""

    # 2. Add 'key="incident_input"' so the text box remembers what was typed
    prompt_text = "Describe the incident in detail:" if lang == "English" else "घटना का विस्तार से वर्णन करें:"
    incident_report = st.text_area(prompt_text, key="incident_input").upper()
    
    submit_text = "🚨 Submit Emergency Report" if lang == "English" else "🚨 आपातकालीन रिपोर्ट दर्ज करें"
    
    # 3. When button is clicked, save the text to memory
    if st.button(submit_text, type="primary"):
        if incident_report:
            st.session_state.saved_report = incident_report
        else:
            st.warning("Please enter a description before submitting." if lang == "English" else "कृपया सबमिट करने से पहले विवरण दर्ज करें।")

    # 4. Display results based on the MEMORY, not the button click
    if st.session_state.saved_report:
        active_report = st.session_state.saved_report
        
        # 1. Belt Tear
        if any(word in active_report for word in ["TEAR", "CUT", "RUPTURE", "BROKEN", "FAT", "TOOT", "FATA", "TUTA"]):
            if lang == "English":
                st.error("🚨 CRITICAL ALERT LOGGED: Belt Tear/Rupture detected.")
                st.error("**Action:** Stop belt immediately. Dispatch vulcanizing crew.")
            else:
                st.error("🚨 गंभीर चेतावनी: बेल्ट फटने की सूचना मिली है।")
                st.error("**कार्रवाई (Action):** तुरंत बेल्ट रोकें। मरम्मत टीम (Vulcanizing crew) को भेजें।")
                
        # 2. Fire/Smoke
        elif any(word in active_report for word in ["FIRE", "SMOKE", "BURNING", "SPARK", "AAG", "DHUAN", "JALA", "SULAG"]):
            if lang == "English":
                st.error("🔥 FIRE EMERGENCY LOGGED: Combustion indicators detected.")
                st.error("**CRITICAL:** Evacuate district. Turn on main suppression systems. Alert DGMS.")
            else:
                st.error("🔥 आग आपातकाल: आग या धुएं की सूचना मिली है।")
                st.error("**खतरा (CRITICAL):** तुरंत खदान खाली करें। वाटर स्प्रिंकलर (Water sprinklers) चालू करें। DGMS को अलर्ट करें।")
                
        # 3. Spillage/Blockage
        elif any(word in active_report for word in ["SPIL", "BLOCK", "OVERFLOW", "JAM", "GIRA", "BHAR", "RUKA", "BAND"]):
            if lang == "English":
                st.warning("⚠️ WARNING LOGGED: Material spillage or blockage reported.")
                st.warning("**Action:** Dispatch cleaning crew to clear idlers and avoid friction fires.")
            else:
                st.warning("⚠️ चेतावनी: कोयला गिरने या बेल्ट जाम होने की सूचना है।")
                st.warning("**कार्रवाई (Action):** सफाई टीम को भेजें ताकि घर्षण (friction) से आग न लगे।")

        # 4. Water/Flooding
        elif any(word in active_report for word in ["WATER", "FLOOD", "INUND", "LEAK", "PAANI", "BAARISH", "RISSA"]):
            if lang == "English":
                st.error("🌊 INUNDATION RISK LOGGED: Water flooding reported.")
                st.error("**Action:** Evacuate immediately. Activate main water pumps. Alert mine manager.")
            else:
                st.error("🌊 बाढ़ का खतरा: खदान में पानी भरने की सूचना है।")
                st.error("**कार्रवाई (Action):** तुरंत बाहर निकलें। मुख्य वाटर पंप चालू करें। खदान प्रबंधक को अलर्ट करें।")
                
        else:
            if lang == "English":
                st.info("📝 General log received. Control room notified for verification.")
            else:
                st.info("📝 रिपोर्ट दर्ज कर ली गई है। वेरिफिकेशन के लिए कंट्रोल रूम को सूचित कर दिया गया है।")
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
        st.info("**🔵 Routine Checks (0-6 Months):**\n* Visual inspection of belt tracking.\n* Check for unusual noise from idlers.")
    elif 6 < machine_age <= 24:
        st.warning("**🟠 Mid-Term Maintenance (6-24 Months):**\n* Lubricate all tail and drive pulleys.\n* Ultrasonic thickness test on belt cover.")
    else:
        st.error("**🔴 Critical Overhaul (>24 Months):**\n* Full structural audit.\n* Replace seized idlers.\n* Test all emergency pull-cords.")
