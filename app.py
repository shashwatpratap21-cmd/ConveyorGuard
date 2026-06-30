import os
import streamlit as st

# FORCE OPENCV HEADLESS MODE VIA ENVIRONMENT VARIABLE
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
from twilio.rest import Client
import datetime
from fpdf import FPDF

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

hourly_loss_lakhs = (capacity * coal_price) / 100000
total_loss_lakhs = hourly_loss_lakhs * downtime

st.sidebar.markdown("---")
st.sidebar.subheader("Estimated Production Loss:")
st.sidebar.error(f"🚨 ₹ {total_loss_lakhs:.2f} Lakh")
st.sidebar.caption("Based on Sijua Colliery average capacity of 600 TPH at ₹2,200/tonne coal price.")
st.sidebar.markdown("---")

st.title("⛏️ ConveyorGuard AI Dashboard")
st.subheader("Tata Steel Unified Multi-Agent Vision System")

# =========================================================================
# --- LOAD YOLOv8 AI AGENTS ---
# =========================================================================
@st.cache_resource
def load_ai_agents():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    conveyor_path = os.path.join(base_dir, 'models', 'conveyor_model.pt')
    spillage_path = os.path.join(base_dir, 'models', 'spillage_model.pt')
    idler_path = os.path.join(base_dir, 'models', 'idler_model.pt')
    
    if not os.path.exists(conveyor_path):
        conveyor_path = os.path.join(base_dir, 'conveyor_model.pt')
        spillage_path = os.path.join(base_dir, 'spillage_model.pt')
        idler_path = os.path.join(base_dir, 'idler_model.pt')

    c_agent = YOLO(conveyor_path)
    s_agent = YOLO(spillage_path)
    i_agent = YOLO(idler_path)
    
    return c_agent, s_agent, i_agent

conveyor_agent = None
spillage_agent = None
idler_agent = None
models_loaded = False
model_error_message = ""

try:
    conveyor_agent, spillage_agent, idler_agent = load_ai_agents()
    models_loaded = True
except Exception as e:
    model_error_message = str(e)
    st.error(f"⚠️ REAL MODEL LOADING ERROR: {model_error_message}")

tab1, tab2, tab3 = st.tabs(["🚨 AI Vision Inspection", "📝 Manual Override (Codes)", "🛠️ Maintenance Scheduler"])

with tab1:
    st.markdown("### Upload Conveyor Belt Image")

    with st.expander("📋 DGMS Pre-Inspection Safety Protocol", expanded=False):
        header_html = '''
        <div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);border-left:4px solid #f39c12;border-radius:8px;padding:20px;margin-bottom:10px;">
            <h4 style="color:#f39c12;margin-top:0;">🟡 CRITICAL UNDERGROUND SAFETY REQUIREMENTS</h4>
            <p style="color:#aaaaaa;font-size:11px;margin-top:-10px;">As per Coal Mines Regulation 2017 & DGMS Circular No. 3 of 2020</p>
        </div>
        '''
        st.markdown(header_html, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            left_html = '''
            <div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:15px;">
                <p style="color:#58a6ff;font-weight:bold;margin-bottom:10px;">📡 Communication</p>
                <p style="color:#e6edf3;font-size:13px;">Inform the <b>surface control room</b> before beginning your inspection walk.</p>
                <hr style="border-color:#30363d;">
                <p style="color:#58a6ff;font-weight:bold;margin-bottom:10px;">📏 Clearance</p>
                <p style="color:#e6edf3;font-size:13px;">Maintain a strict <b>1.5m clearance</b> from moving idlers, tail pulleys, and the drive head at all times.</p>
            </div>
            '''
            st.markdown(left_html, unsafe_allow_html=True)

        with col_b:
            right_html = '''
            <div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:15px;">
                <p style="color:#58a6ff;font-weight:bold;margin-bottom:10px;">🚷 Movement</p>
                <p style="color:#e6edf3;font-size:13px;"><b>NEVER</b> step over, under, or onto a moving belt. Use designated <b>crossover bridges only</b>.</p>
                <hr style="border-color:#30363d;">
                <p style="color:#58a6ff;font-weight:bold;margin-bottom:10px;">🆘 Emergency Readiness</p>
                <p style="color:#e6edf3;font-size:13px;">Visually locate the nearest <b>emergency pull-cord</b> before beginning your inspection.</p>
            </div>
            '''
            st.markdown(right_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        footer_html = '''
        <div style="background:linear-gradient(135deg,#2d1b1b 0%,#1a0a0a 100%);border:1px solid #f85149;border-radius:8px;padding:12px 20px;text-align:center;">
            <span style="color:#f85149;font-weight:bold;font-size:13px;">
                🔴 EMERGENCY PROTOCOL &nbsp;|&nbsp; Pull cord → Report to Overman → Evacuate → Alert Surface Control Room
            </span>
        </div>
        '''
        st.markdown(footer_html, unsafe_allow_html=True)

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
        if not models_loaded or conveyor_agent is None:
            st.error("🚨 The AI models failed to load in the background. We cannot process the image. Here is the exact reason why:")
            st.code(model_error_message)
            st.stop()

        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image)

        col_img, col_results = st.columns([1.5, 1])

        with col_img:
            with st.spinner("AI Agents inspecting conveyor belt..."):
                res_conveyor = conveyor_agent(img_array, conf=confidence_threshold, verbose=False)
                res_spillage = spillage_agent(img_array, conf=confidence_threshold, verbose=False)
                res_idler = idler_agent(img_array, conf=confidence_threshold, verbose=False)

                annotated_img = res_conveyor[0].plot()
                annotated_img = res_spillage[0].plot(img=annotated_img)
                annotated_img = res_idler[0].plot(img=annotated_img)

                final_display_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

            st.image(final_display_img, use_container_width=True)

        with col_results:
            st.markdown("### Inspection Verdict:")

            def count_detections(result):
                try:
                    if result[0].masks is not None:
                        return len(result[0].masks)
                    elif result[0].boxes is not None:
                        return len(result[0].boxes)
                    return 0
                except:
                    return 0

            total_anomalies = (
                count_detections(res_conveyor) +
                count_detections(res_spillage) +
                count_detections(res_idler)
            )

            if total_anomalies > 0:
                st.error(f"🚨 {total_anomalies} ANOMALIES DETECTED")
                st.error("**Action:** Dispatch maintenance team to verify zones.")
            else:
                st.success("✅ NORMAL / HEALTHY LOAD")
                st.success("AI detected zero anomalies above the threshold.")

# --- TAB 2: MANUAL OVERRIDE ---
with tab2:
    st.markdown("### 🎙️ Emergency Manual Reporting")
    st.write("Use your device's **Microphone (Dictation)**. Regional languages are supported.")
    
    lang = st.radio("Response
