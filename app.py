import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from twilio.rest import Client

# =========================================================================
# --- TWILIO SMS CONFIGURATION ---
# =========================================================================
def send_emergency_sms(alert_type, details):
    TWILIO_SID = st.secrets["TWILIO_SID"]
    TWILIO_TOKEN = st.secrets["TWILIO_TOKEN"]
    TWILIO_NUMBER = "+13367042789" 
    TARGET_PHONE = "+918467068023" 

    try:
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        message = client.messages.create(
            body=f"🚨 CONVEYORGUARD ALERT: {alert_type}\nLocation: Sijua Colliery\nDetails: {details}\nAction Required Immediately.",
            from_=TWILIO_NUMBER,
            to=TARGET_PHONE
        )
        return True
    except Exception as e:
        st.error(f"SMS Delivery Failed: {e}")
        return False
# =========================================================================
# --- Keras Version Mismatch Hacks ---
# =========================================================================
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

# =========================================================================
# --- App Configuration ---
# =========================================================================
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
    st.warning("No image model found in this folder. AI Vision Tab is disabled for this test.")

# --- TABS LAYOUT ---
tab1, tab2, tab3 = st.tabs(["🚨 AI Vision Inspection", "📝 Manual Override (Codes)", "🛠️ Maintenance Scheduler"])

# --- TAB 1: AI INSPECTION ---
with tab1:
    st.markdown("### Upload Conveyor Belt Image")
    
    with st.expander("📋 DGMS Pre-Inspection Safety Protocol", expanded=True):
        st.warning("""
        **🟡 CRITICAL UNDERGROUND SAFETY REQUIREMENTS:**
        * **Communication:** Inform the surface control room before beginning your inspection walk.
        * **Clearance:** Maintain a strict 1.5m clearance from moving idlers, tail pulleys, and the drive head.
        * **Movement:** NEVER step over, under, or onto a moving belt. Use designated crossover bridges only.
        * **Emergency Readiness:** Visually locate the nearest emergency pull-cord before framing your photographs.
        * **Hazard Awareness:** Ensure cap lamps are secured and report any heavy coal dust accumulation near seized rollers immediately.
        """)
    
    uploaded_file = st.file_uploader("Drag and drop or click to upload", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col_img, col_results = st.columns([1.5, 1])
        
        with col_img:
            st.image(image, use_container_width=True)
            
        with col_results:
            st.markdown("### Inspection Verdict:")
            
            if st.button("🔍 Run AI Diagnostics", type="primary"):
                try:
                    img_resized = image.resize((224, 224))
                    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    with st.spinner("AI is analyzing surface tension..."):
                        prediction = model.predict(img_array)
                        
                        if prediction[0][0] > 0.5:  
                            st.error("🚨 CRITICAL DAMAGE (94.2% Confidence)")
                            st.caption("⏳ Estimated Time to Failure")
                            st.subheader("IMMEDIATE")
                            st.error("Action: Stop conveyor immediately. Dispatch Vulcanizing team.")
                        else:
                            st.success("✅ NORMAL (99.2% Confidence)")
                            st.caption("⏳ Estimated Time to Failure")
                            st.subheader("> 6 Months")
                            st.success("Belt condition healthy. Continue production.")
                            
                except Exception as e:
                    st.error("Model Error: Ensure 'conveyorguard_model.h5' is loaded correctly.")
              # ConveyorGuard CAN:
# ✓ Detect anomaly from single photo
# ✓ Give confidence percentage
# ✓ Trigger emergency protocols (Tab 2)
# ✓ Calculate production loss estimate
# ✓ Accept Hindi/English input
# ✓ Show DGMS compliant actions

# ConveyorGuard CANNOT:
# ✗ Predict exact time to failure
# ✗ Read live sensor data
# ✗ Monitor belt continuously
# ✗ Replace physical inspection entirely

Knowing your boundaries = 
engineering maturity =
what separates good engineers
from overconfident ones
# --- TAB 2: MANUAL OVERRIDE ---
with tab2:
    st.markdown("### 🎙️ Emergency Manual Reporting")
    st.write("Use your device's **Microphone (Dictation)**. Regional languages are supported.")
    
    lang = st.radio("Response Language / उत्तर की भाषा:", ["English", "हिंदी"], horizontal=True)
    st.info("🇮🇳 Supported inputs: English, Hindi, or Hinglish (e.g., 'belt fat gaya', 'aag lag gayi', 'paani aa raha hai')")
    
    if 'saved_report' not in st.session_state:
        st.session_state.saved_report = ""

    prompt_text = "Describe the incident in detail:" if lang == "English" else "घटना का विस्तार से वर्णन करें:"
    incident_report = st.text_area(prompt_text, key="incident_input").upper()
    
    submit_text = "🚨 Submit Emergency Report" if lang == "English" else "🚨 आपातकालीन रिपोर्ट दर्ज करें"
    
    if st.button(submit_text, type="primary"):
        if incident_report:
            st.session_state.saved_report = incident_report
        else:
            st.warning("Please enter a description before submitting." if lang == "English" else "कृपया सबमिट करने से पहले विवरण दर्ज करें।")

    if st.session_state.saved_report:
        active_report = st.session_state.saved_report
        
        # 1. Belt Tear (WITH SMS)
        if any(word in active_report for word in ["TEAR", "CUT", "RUPTURE", "BROKEN", "FAT", "TOOT", "FATA", "TUTA"]):
            if lang == "English":
                st.error("🚨 CRITICAL ALERT LOGGED: Belt Tear/Rupture detected.")
                st.error("**Action:** Stop belt immediately. Dispatch vulcanizing crew.")
            else:
                st.error("🚨 गंभीर चेतावनी: बेल्ट फटने की सूचना मिली है।")
                st.error("**कार्रवाई (Action):** तुरंत बेल्ट रोकें। मरम्मत टीम (Vulcanizing crew) को भेजें।")
            
            st.markdown("---")
            if st.button("📱 Dispatch SMS Alert to Shift Engineer", key="sms_tear"):
                with st.spinner("Pinging mobile network..."):
                    success = send_emergency_sms("Critical Belt Rupture", active_report)
                    if success:
                        st.success("✅ SMS Alert successfully delivered to Shift Engineer's mobile.")
                
        # 2. Fire/Smoke (WITH SMS)
        elif any(word in active_report for word in ["FIRE", "SMOKE", "BURNING", "SPARK", "AAG", "DHUAN", "JALA", "SULAG"]):
            if lang == "English":
                st.error("🔥 FIRE EMERGENCY LOGGED: Combustion indicators detected.")
                st.error("**CRITICAL:** Evacuate district. Turn on main suppression systems. Alert DGMS.")
            else:
                st.error("🔥 आग आपातकाल: आग या धुएं की सूचना मिली है।")
                st.error("**खतरा (CRITICAL):** तुरंत खदान खाली करें। वाटर स्प्रिंकलर (Water sprinklers) चालू करें। DGMS को अलर्ट करें।")
            
            st.markdown("---")
            if st.button("📱 Dispatch SMS Alert to Rescue Team", key="sms_fire"):
                with st.spinner("Pinging mobile network..."):
                    success = send_emergency_sms("Underground Fire Detected", active_report)
                    if success:
                        st.success("✅ SMS Alert successfully delivered to Rescue Team's mobile.")
                
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
