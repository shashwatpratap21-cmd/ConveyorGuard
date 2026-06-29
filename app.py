import streamlit as st
import os
# We do NOT import cv2 or ultralytics until AFTER setting environment variables
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
from twilio.rest import Client
# ... rest of your code
import datetime
from fpdf import FPDF

# FORCE OPENCV HEADLESS MODE VIA ENVIRONMENT VARIABLE
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# ... [Keep the rest of your app.py code exactly the same below this] ...
# =========================================================================
# --- TWILIO SMS CONFIGURATION ---
# =========================================================================
def send_emergency_sms(alert_type, details):
    TWILIO_SID = st.secrets.get("TWILIO_SID", "Dummy_SID")
    TWILIO_TOKEN = st.secrets.get("TWILIO_TOKEN", "Dummy_Token")
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
        st.error(f"SMS Delivery Failed: Check Twilio Secrets or Network. Error: {e}")
        return False

# =========================================================================
# --- App Configuration ---
# =========================================================================
st.set_page_config(
    page_title="ConveyorGuard Vision", 
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
st.title("⛏️ ConveyorGuard AI Dashboard")
st.subheader("Tata Steel Unified Multi-Agent Vision System")

# =========================================================================
# --- LOAD YOLOv8 AI AGENTS ---
# =========================================================================
@st.cache_resource
def load_ai_agents():
    # This automatically finds the exact folder app.py is sitting in
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try looking for 'models/file.pt' first
    conveyor_path = os.path.join(base_dir, 'models', 'conveyor_model.pt')
    spillage_path = os.path.join(base_dir, 'models', 'spillage_model.pt')
    idler_path = os.path.join(base_dir, 'models', 'idler_model.pt')
    
    # Fallback: If 'models/' folder isn't found, look directly in the main folder
    if not os.path.exists(conveyor_path):
        conveyor_path = os.path.join(base_dir, 'conveyor_model.pt')
        spillage_path = os.path.join(base_dir, 'spillage_model.pt')
        idler_path = os.path.join(base_dir, 'idler_model.pt')
        
    print(f"Loading weights from paths:\n{conveyor_path}\n{spillage_path}\n{idler_path}")
    
    conveyor_agent = YOLO(conveyor_path)
    spillage_agent = YOLO(spillage_path)
    idler_agent = YOLO(idler_path)
    
    return conveyor_agent, spillage_agent, idler_agent
try:
    conveyor_agent, spillage_agent, idler_agent = load_ai_agents()
except Exception as e:
    st.warning(f"Could not load AI models. Ensure 'models' folder exists with the .pt files. Error: {e}")

# --- TABS LAYOUT ---
tab1, tab2, tab3 = st.tabs(["🚨 AI Vision Inspection", "📝 Manual Override (Codes)", "🛠️ Maintenance Scheduler"])

# --- TAB 1: AI INSPECTION ---
with tab1:
    st.markdown("### Upload Conveyor Belt Image")
    
    with st.expander("📋 DGMS Pre-Inspection Safety Protocol", expanded=False):
        st.warning("""
        **🟡 CRITICAL UNDERGROUND SAFETY REQUIREMENTS:**
        * **Communication:** Inform the surface control room before beginning your inspection walk.
        * **Clearance:** Maintain a strict 1.5m clearance from moving idlers, tail pulleys, and the drive head.
        """)
    
    # --- ENTERPRISE UPGRADE: Admin Locked Calibration ---
    st.markdown("### 🎛️ AI Calibration (Safety Officer Only)")
    
    admin_password = st.text_input("Enter Admin Password to Unlock Calibration:", type="password")
    
    if admin_password == "dgms2026": 
        st.success("🔓 Calibration Unlocked")
        confidence_threshold = st.slider("Detection Confidence Threshold", 0.10, 0.99, 0.25, 0.01)
    else:
        st.info("🔒 System running at statutory default threshold (0.25).")
        confidence_threshold = 0.25  
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Drag and drop or click to upload", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Convert uploaded image to OpenCV format
        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image)
        
        col_img, col_results = st.columns([1.5, 1])
        
        with col_img:
            with st.spinner("AI Agents inspecting conveyor belt..."):
                # Run Inference across all 3 models
                res_conveyor = conveyor_agent(img_array, conf=confidence_threshold, verbose=False)
                res_spillage = spillage_agent(img_array, conf=confidence_threshold, verbose=False)
                res_idler = idler_agent(img_array, conf=confidence_threshold, verbose=False)

                # Layer the bounding boxes on top of each other
                annotated_img = res_conveyor[0].plot()
                annotated_img = res_spillage[0].plot(img=annotated_img)
                annotated_img = res_idler[0].plot(img=annotated_img)
                
                # YOLO plots in BGR, convert back to RGB for Streamlit
                final_display_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                
            st.image(final_display_img, use_container_width=True)
            
        with col_results:
            st.markdown("### Inspection Verdict:")
            
            # Count total detections
            total_anomalies = len(res_conveyor[0].boxes) + len(res_spillage[0].boxes) + len(res_idler[0].boxes)
            
            if total_anomalies > 0:  
                st.error(f"🚨 {total_anomalies} ANOMALIES DETECTED")
                st.error("**Action:** Dispatch maintenance team to verify bounding box zones.")
                st.markdown("---")
                
                # Dynamic Breakdown
                if len(res_conveyor[0].boxes) > 0: st.warning("⚠️ Edge alignment or debris detected.")
                if len(res_spillage[0].boxes) > 0: st.warning("⚠️ Material spillage or foreign objects detected.")
                if len(res_idler[0].boxes) > 0: st.warning("⚠️ Idler/Roller anomaly detected.")
                
                st.info("""
                **📋 DGMS Statutory Recommendation:**
                * Immediate physical inspection required.
                * Log incident in the statutory register.
                """)
            else:
                st.success(f"✅ NORMAL / HEALTHY LOAD")
                st.success("AI detected zero anomalies above the threshold.")
                st.markdown("---")
                st.info("""
                **📋 Routine Recommendation:**
                * Next scheduled inspection: 7 days
                * Standard: DGMS Circular No. 3 of 2020
                """)

# --- TAB 2: MANUAL OVERRIDE ---
with tab2:
    st.markdown("### 🎙️ Emergency Manual Reporting")
    st.write("Use your device's **Microphone (Dictation)**. Regional languages are supported.")
    
    lang = st.radio("Response Language / उत्तर की भाषा:", ["English", "हिंदी"], horizontal=True)
    
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
        
        # Incident Checks
        if any(word in active_report for word in ["TEAR", "CUT", "BROKEN", "FATA", "TUTA"]):
            st.error("🚨 CRITICAL ALERT LOGGED: Belt Tear/Rupture detected." if lang == "English" else "🚨 गंभीर चेतावनी: बेल्ट फटने की सूचना मिली है।")
            if st.button("🚨 Dispatch Alert Network", type="primary"):
                send_emergency_sms("Critical Belt Rupture", active_report)
                
        elif any(word in active_report for word in ["FIRE", "SMOKE", "AAG", "DHUAN"]):
            st.error("🔥 FIRE EMERGENCY LOGGED: Combustion indicators detected." if lang == "English" else "🔥 आग आपातकाल: आग या धुएं की सूचना मिली है।")
            if st.button("🚨 Dispatch Alert Network", type="primary"):
                send_emergency_sms("Underground Fire Detected", active_report)
                
        elif any(word in active_report for word in ["WATER", "FLOOD", "PAANI", "BAARISH"]):
            st.error("🌊 INUNDATION RISK LOGGED: Water flooding reported." if lang == "English" else "🌊 बाढ़ का खतरा: खदान में पानी भरने की सूचना है।")
            if st.button("🚨 Dispatch Alert Network", type="primary"):
                send_emergency_sms("Critical Inundation Risk", active_report)
        else:
            st.info("📝 General log received. Control room notified." if lang == "English" else "📝 रिपोर्ट दर्ज कर ली गई है।")
        
        # PDF Generator
        st.markdown("---")
        if st.button("📥 Generate Statutory PDF" if lang == "English" else "📥 वैधानिक PDF जनरेट करें"):
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(200, 10, "CONVEYORGUARD - DGMS STATUTORY LOG", ln=True, align='C')
            pdf.set_font("Arial", "", 12)
            clean_report = active_report.replace('\n', ' ').encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 10, clean_report)
            pdf_bytes = pdf.output(dest="S").encode("latin-1")
            
            st.download_button(
                label="📄 Download Official DGMS Report (PDF)" if lang == "English" else "📄 आधिकारिक DGMS रिपोर्ट डाउनलोड करें (PDF)",
                data=pdf_bytes,
                file_name=f"DGMS_Report_{current_time[:10]}.pdf",
                mime="application/pdf",
                type="primary"
            )

# --- TAB 3: MAINTENANCE SCHEDULER ---
with tab3:
    st.markdown("### 🛠️ Predictive Maintenance & Statutory Compliance")
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Next Statutory Walkthrough", value="2 Days", delta="-1 day (Urgent)", delta_color="inverse")
    col2.metric(label="Idler Greasing Status", value="Overdue", delta="Action Req", delta_color="inverse")
    col3.metric(label="Belt Tension & Alignment", value="Normal", delta="14 Days left", delta_color="normal")
