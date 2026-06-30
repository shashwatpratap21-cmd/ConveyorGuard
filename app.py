import os
import streamlit as st

# OpenCV cloud setting
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
        client.messages.create(
            body=f"🚨 CONVEYORGUARD ALERT: {alert_type}\nLocation: Sijua Colliery\nDetails: {details}\nAction Required Immediately.",
            from_=TWILIO_NUMBER,
            to=TARGET_PHONE
        )
        return True
    except Exception as e:
        st.error(f"SMS Delivery Failed: Check Twilio Secrets or Network. Error: {e}")
        return False


# =========================================================================
# --- APP CONFIGURATION ---
# =========================================================================

st.set_page_config(
    page_title="ConveyorGuard Vision",
    page_icon="⛏️",
    layout="wide"
)

st.sidebar.header("📉 Economic Impact Calculator")
st.sidebar.markdown("Estimate financial loss during downtime.")

capacity = st.sidebar.number_input(
    "Conveyor Capacity (TPH)",
    min_value=100,
    max_value=5000,
    value=600,
    step=50
)

coal_price = st.sidebar.number_input(
    "Coal Price (₹/t)",
    min_value=1000,
    max_value=10000,
    value=2200,
    step=100
)

downtime = st.sidebar.number_input(
    "Predicted Downtime (h)",
    min_value=0.5,
    max_value=24.0,
    value=3.0,
    step=0.5
)

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

    conveyor_path = os.path.join(base_dir, "models", "conveyor_model.pt")
    spillage_path = os.path.join(base_dir, "models", "spillage_model.pt")
    idler_path = os.path.join(base_dir, "models", "idler_model.pt")

    # fallback if models are uploaded directly in root folder
    if not os.path.exists(conveyor_path):
        conveyor_path = os.path.join(base_dir, "conveyor_model.pt")
        spillage_path = os.path.join(base_dir, "spillage_model.pt")
        idler_path = os.path.join(base_dir, "idler_model.pt")

    conveyor_agent = YOLO(conveyor_path)
    spillage_agent = YOLO(spillage_path)
    idler_agent = YOLO(idler_path)

    return conveyor_agent, spillage_agent, idler_agent


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


# =========================================================================
# --- HELPER FUNCTIONS ---
# =========================================================================

def count_detections(result):
    try:
        if result is None:
            return 0

        if result[0].masks is not None:
            return len(result[0].masks)

        if result[0].boxes is not None:
            return len(result[0].boxes)

        return 0
    except Exception:
        return 0


def run_single_model(model, image_array, confidence):
    result = model(image_array, conf=confidence, verbose=False)
    annotated = result[0].plot()
    return result, annotated


# =========================================================================
# --- TABS ---
# =========================================================================

tab1, tab2, tab3 = st.tabs([
    "🚨 AI Vision Inspection",
    "📝 Manual Override (Codes)",
    "🛠️ Maintenance Scheduler"
])


# =========================================================================
# --- TAB 1: AI VISION INSPECTION ---
# =========================================================================

with tab1:
    st.markdown("### Upload Conveyor Belt Image")

    with st.expander("📋 DGMS Pre-Inspection Safety Protocol", expanded=False):
        st.markdown(
            """
            <div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
            border-left:4px solid #f39c12;border-radius:8px;padding:20px;margin-bottom:10px;">
                <h4 style="color:#f39c12;margin-top:0;">🟡 CRITICAL UNDERGROUND SAFETY REQUIREMENTS</h4>
                <p style="color:#aaaaaa;font-size:11px;margin-top:-10px;">
                As per Coal Mines Regulation 2017 & DGMS Circular No. 3 of 2020</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(
                """
                <div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:15px;">
                    <p style="color:#58a6ff;font-weight:bold;margin-bottom:10px;">📡 Communication</p>
                    <p style="color:#e6edf3;font-size:13px;">
                    Inform the <b>surface control room</b> before beginning your inspection walk.</p>
                    <hr style="border-color:#30363d;">
                    <p style="color:#58a6ff;font-weight:bold;margin-bottom:10px;">📏 Clearance</p>
                    <p style="color:#e6edf3;font-size:13px;">
                    Maintain a strict <b>1.5m clearance</b> from moving idlers, tail pulleys, and the drive head.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_b:
            st.markdown(
                """
                <div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:15px;">
                    <p style="color:#58a6ff;font-weight:bold;margin-bottom:10px;">🚷 Movement</p>
                    <p style="color:#e6edf3;font-size:13px;">
                    <b>NEVER</b> step over, under, or onto a moving belt. Use designated crossover bridges only.</p>
                    <hr style="border-color:#30363d;">
                    <p style="color:#58a6ff;font-weight:bold;margin-bottom:10px;">🆘 Emergency Readiness</p>
                    <p style="color:#e6edf3;font-size:13px;">
                    Visually locate the nearest <b>emergency pull-cord</b> before beginning your inspection.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            """
            <div style="background:linear-gradient(135deg,#2d1b1b 0%,#1a0a0a 100%);
            border:1px solid #f85149;border-radius:8px;padding:12px 20px;text-align:center;">
                <span style="color:#f85149;font-weight:bold;font-size:13px;">
                    🔴 EMERGENCY PROTOCOL | Pull cord → Report to Overman → Evacuate → Alert Surface Control Room
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 🎛️ AI Calibration")

    admin_password = st.text_input(
        "Enter Admin Password to Unlock Calibration:",
        type="password"
    )

    if admin_password == "dgms2026":
        st.success("🔓 Calibration Unlocked")
        confidence_threshold = st.slider(
            "Detection Confidence Threshold",
            0.10,
            0.99,
            0.25,
            0.01
        )
    else:
        st.info("🔒 System running at statutory default threshold (0.25).")
        confidence_threshold = 0.25

    st.markdown("### 🧠 Select Inspection View")

    inspection_mode = st.selectbox(
        "Choose the type of image you are uploading:",
        [
            "Belt Surface Damage",
            "Spillage / Foreign Object",
            "Idler / Roller Mechanical Health",
            "Full Multi-Agent Scan"
        ]
    )

    st.caption(
        "Use Full Multi-Agent Scan only for final demo. For normal testing, select the specific view to reduce false detections."
    )

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Drag and drop or click to upload",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        if not models_loaded or conveyor_agent is None:
            st.error("🚨 The AI models failed to load. Reason:")
            st.code(model_error_message)
            st.stop()

        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        col_img, col_results = st.columns([1.5, 1])

        with col_img:
            with st.spinner("AI Agents inspecting conveyor belt..."):

                res_conveyor = None
                res_spillage = None
                res_idler = None

                if inspection_mode == "Belt Surface Damage":
                    res_conveyor, annotated_img = run_single_model(
                        conveyor_agent,
                        img_array,
                        confidence_threshold
                    )

                elif inspection_mode == "Spillage / Foreign Object":
                    res_spillage, annotated_img = run_single_model(
                        spillage_agent,
                        img_array,
                        max(confidence_threshold, 0.55)
                    )

                elif inspection_mode == "Idler / Roller Mechanical Health":
                    res_idler, annotated_img = run_single_model(
                        idler_agent,
                        img_array,
                        max(confidence_threshold, 0.60)
                    )

                else:
                    res_conveyor = conveyor_agent(
                        img_array,
                        conf=confidence_threshold,
                        verbose=False
                    )

                    res_spillage = spillage_agent(
                        img_array,
                        conf=max(confidence_threshold, 0.55),
                        verbose=False
                    )

                    res_idler = idler_agent(
                        img_array,
                        conf=max(confidence_threshold, 0.60),
                        verbose=False
                    )

                    annotated_img = res_conveyor[0].plot()
                    annotated_img = res_spillage[0].plot(img=annotated_img)
                    annotated_img = res_idler[0].plot(img=annotated_img)

                final_display_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

            st.image(final_display_img, use_container_width=True)

        with col_results:
            st.markdown("### Inspection Verdict:")
            with st.expander("🔍 Debug: Model Raw Output"):
                if res_conveyor is not None:
                    st.write("Conveyor model classes:", res_conveyor[0].names)
                    st.write("Conveyor boxes:", len(res_conveyor[0].boxes) if res_conveyor[0].boxes is not None else 0)
                    st.write("Conveyor masks:", len(res_conveyor[0].masks) if res_conveyor[0].masks is not None else 0)
                    if res_spillage is not None:
                        st.write("Spillage model classes:", res_spillage[0].names)
                        st.write("Spillage boxes:", len(res_spillage[0].boxes) if res_spillage[0].boxes is not None else 0)
                        if res_idler is not None:
        st.write("Idler model classes:", res_idler[0].names)
        st.write("Idler boxes:", len(res_idler[0].boxes) if res_idler[0].boxes is not None else 0)
            conveyor_count = count_detections(res_conveyor)
            spillage_count = count_detections(res_spillage)
            idler_count = count_detections(res_idler)

            total_anomalies = conveyor_count + spillage_count + idler_count

            if total_anomalies > 0:
                st.error(f"🚨 {total_anomalies} ANOMALIES DETECTED")
                st.error("**Action:** Dispatch maintenance team to verify zones.")

                if conveyor_count > 0:
                    st.warning("⚠️ Belt surface damage / tear / hole detected.")

                if spillage_count > 0:
                    st.warning("⚠️ Spillage or foreign object detected.")

                if idler_count > 0:
                    st.warning("⚠️ Idler / roller / mechanical anomaly detected.")

                st.info(
                    "📋 DGMS Recommendation:\n"
                    "- Immediate physical inspection required.\n"
                    "- Log incident in statutory register.\n"
                    "- Stop conveyor if defect is severe."
                )

            else:
                st.success("✅ NORMAL / HEALTHY LOAD")
                st.success("AI detected zero anomalies above the threshold.")
                st.info(
                    "📋 Routine Recommendation:\n"
                    "- Continue monitoring.\n"
                    "- Next scheduled inspection: 7 days.\n"
                    "- Follow DGMS Circular No. 3 of 2020."
                )


# =========================================================================
# --- TAB 2: MANUAL OVERRIDE ---
# =========================================================================

with tab2:
    st.markdown("### 🎙️ Emergency Manual Reporting")
    st.write("Use your device's microphone/dictation. English, Hindi, and Hinglish are supported.")

    lang = st.radio(
        "Response Language / उत्तर की भाषा:",
        ["English", "हिंदी"],
        horizontal=True
    )

    if "saved_report" not in st.session_state:
        st.session_state["saved_report"] = ""

    prompt_text = "Describe the incident in detail:" if lang == "English" else "घटना का विस्तार से वर्णन करें:"
    incident_report = st.text_area(prompt_text, key="incident_input").upper()

    submit_text = "🚨 Submit Emergency Report" if lang == "English" else "🚨 आपातकालीन रिपोर्ट दर्ज करें"

    if st.button(submit_text, type="primary"):
        if incident_report:
            st.session_state["saved_report"] = incident_report
        else:
            st.warning(
                "Please enter a description before submitting."
                if lang == "English"
                else "कृपया सबमिट करने से पहले विवरण दर्ज करें।"
            )

    if st.session_state["saved_report"]:
        active_report = st.session_state["saved_report"]

        if any(word in active_report for word in ["TEAR", "CUT", "BROKEN", "FATA", "TUTA"]):
            st.error(
                "🚨 CRITICAL ALERT LOGGED: Belt Tear/Rupture detected."
                if lang == "English"
                else "🚨 गंभीर चेतावनी: बेल्ट फटने की सूचना मिली है।"
            )

            if st.button("📱 Dispatch SMS Alert - Belt Tear"):
                send_emergency_sms("Critical Belt Rupture", active_report)

        elif any(word in active_report for word in ["FIRE", "SMOKE", "AAG", "DHUAN"]):
            st.error(
                "🔥 FIRE EMERGENCY LOGGED: Combustion indicators detected."
                if lang == "English"
                else "🔥 आग आपातकाल: आग या धुएं की सूचना मिली है।"
            )

            if st.button("📱 Dispatch SMS Alert - Fire"):
                send_emergency_sms("Underground Fire Detected", active_report)

        elif any(word in active_report for word in ["WATER", "FLOOD", "PAANI", "BAARISH"]):
            st.error(
                "🌊 INUNDATION RISK LOGGED: Water flooding reported."
                if lang == "English"
                else "🌊 बाढ़ का खतरा: खदान में पानी भरने की सूचना है।"
            )

            if st.button("📱 Dispatch SMS Alert - Water"):
                send_emergency_sms("Critical Inundation Risk", active_report)

        elif any(word in active_report for word in ["SPILL", "BLOCK", "JAM", "GARBAGE", "IRON"]):
            st.warning(
                "⚠️ WARNING LOGGED: Spillage / blockage / foreign object reported."
                if lang == "English"
                else "⚠️ चेतावनी: स्पिलेज / ब्लॉकेज / विदेशी वस्तु की सूचना मिली है।"
            )

        else:
            st.info(
                "📝 General log received. Control room notified."
                if lang == "English"
                else "📝 रिपोर्ट दर्ज कर ली गई है।"
            )

        st.markdown("---")

        if st.button(
            "📥 Generate Statutory PDF"
            if lang == "English"
            else "📥 वैधानिक PDF जनरेट करें"
        ):
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(200, 10, "CONVEYORGUARD - DGMS STATUTORY LOG", ln=True, align="C")

            pdf.set_font("Arial", "", 12)
            clean_report = active_report.replace("\n", " ").encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(0, 10, clean_report)

            pdf_bytes = pdf.output(dest="S").encode("latin-1")

            st.download_button(
                label="📄 Download Official DGMS Report (PDF)"
                if lang == "English"
                else "📄 आधिकारिक DGMS रिपोर्ट डाउनलोड करें (PDF)",
                data=pdf_bytes,
                file_name=f"DGMS_Report_{current_time[:10]}.pdf",
                mime="application/pdf",
                type="primary"
            )


# =========================================================================
# --- TAB 3: MAINTENANCE SCHEDULER ---
# =========================================================================

with tab3:
    st.markdown("### 🛠️ Predictive Maintenance & Statutory Compliance")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        label="Next Statutory Walkthrough",
        value="2 Days",
        delta="-1 day (Urgent)",
        delta_color="inverse"
    )

    col2.metric(
        label="Idler Greasing Status",
        value="Overdue",
        delta="Action Req",
        delta_color="inverse"
    )

    col3.metric(
        label="Belt Tension & Alignment",
        value="Normal",
        delta="14 Days left",
        delta_color="normal"
    )

    st.markdown("---")

    st.markdown("#### 📋 DGMS Compliance Tracker")
    st.checkbox("Weekly Belt Inspection", value=True, disabled=True)
    st.checkbox("Emergency Pull Cord Test", value=True, disabled=True)
    st.checkbox("Fire Extinguisher & Sprinkler Check", value=False, disabled=True)
    st.checkbox("Walkthrough Record Updated", value=True, disabled=True)
