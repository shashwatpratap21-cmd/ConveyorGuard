import os
import datetime

import streamlit as st

# OpenCV cloud setting for Streamlit Cloud
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import cv2
import numpy as np
from PIL import Image
from fpdf import FPDF
from twilio.rest import Client
from ultralytics import YOLO


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
            body=(
                f"CONVEYORGUARD ALERT: {alert_type}\n"
                f"Location: Sijua Colliery\n"
                f"Details: {details}\n"
                f"Action Required Immediately."
            ),
            from_=TWILIO_NUMBER,
            to=TARGET_PHONE,
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
    layout="wide",
)

st.sidebar.header("📉 Economic Impact Calculator")
st.sidebar.markdown("Estimate financial loss during downtime.")

capacity = st.sidebar.number_input(
    "Conveyor Capacity (TPH)", min_value=100, max_value=5000, value=600, step=50
)
coal_price = st.sidebar.number_input(
    "Coal Price (₹/t)", min_value=1000, max_value=10000, value=2200, step=100
)
downtime = st.sidebar.number_input(
    "Predicted Downtime (h)", min_value=0.5, max_value=24.0, value=3.0, step=0.5
)

hourly_loss_lakhs = (capacity * coal_price) / 100000
total_loss_lakhs = hourly_loss_lakhs * downtime

st.sidebar.markdown("---")
st.sidebar.subheader("Estimated Production Loss:")
st.sidebar.error(f"🚨 ₹ {total_loss_lakhs:.2f} Lakh")
st.sidebar.caption(
    "Based on Sijua Colliery average capacity of 600 TPH at ₹2,200/tonne coal price."
)
st.sidebar.markdown("---")

st.title("⛏️ ConveyorGuard AI Dashboard")
st.subheader("Tata Steel Unified Multi-Agent Vision System")


# =========================================================================
# --- LOAD YOLOv8 AI AGENTS ---
# =========================================================================

@st.cache_resource
def load_ai_agents():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    conveyor_path = os.path.join(base_dir, "conveyor_model.pt")
    spillage_path = os.path.join(base_dir, "spillage_model.pt")
    idler_path = os.path.join(base_dir, "idler_model.pt")

    missing = []

    for path in [conveyor_path, spillage_path, idler_path]:
        if not os.path.exists(path):
            missing.append(path)

    if missing:
        raise FileNotFoundError("Missing model files: " + ", ".join(missing))

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
# --- FILTERING AND DRAWING HELPERS ---
# =========================================================================

# Belt surface model classes from your debug:
# 0: Hole
# 1: Human
# 2: Other Objects
# 3: Puncture
# 4: Roller
# 5: Tear
# 6: impact damage
# 7: patch work

BELT_ALLOWED = [
    "hole",
    "puncture",
    "tear",
    "impact damage",
    "impact",
    "patch work",
    "patch",
    "damage",
    "surface_damage",
    "surface damage",
    "defaut",
    "default",
    "crack",
    "fissure",
    "burn",
    "brulure",
    "wear",
    "abrasion"
]

BELT_BLOCKED = [
    "human",
    "person",
    "roller",
    "idler",
    "pole",
    "other objects",
    "other object",
    "coal",
    "rock",
    "load"
]

SPILLAGE_ALLOWED = [
    "garbage",
    "block",
    "iron",
    "brazing",
    "foreign",
    "foreign object",
    "spillage",
    "debris"
]

SPILLAGE_BLOCKED = [
    "person",
    "human",
    "roller",
    "idler",
    "pole",
    "other objects",
    "other object"
]

IDLER_ALLOWED = [
    "idler",
    "roller",
    "pole",
    "missing",
    "damaged",
    "damage",
    "misalignment",
    "desalineamiento"
]

IDLER_BLOCKED = [
    "person",
    "human"
]


def model_names(model):
    try:
        names = model.names

        if isinstance(names, dict):
            return names

        return {i: name for i, name in enumerate(names)}

    except Exception:
        return {}


def class_is_allowed(class_name, allowed_keywords=None, blocked_keywords=None):
    name = class_name.lower().strip()

    if blocked_keywords:
        if any(word in name for word in blocked_keywords):
            return False

    if allowed_keywords:
        return any(word in name for word in allowed_keywords)

    return True


def draw_label(img, text, x1, y1, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    # Correct OpenCV text size syntax
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    y1 = max(y1, th + 10)

    cv2.rectangle(
        img,
        (x1, y1 - th - 10),
        (x1 + tw + 10, y1),
        color,
        -1
    )

    cv2.putText(
        img,
        text,
        (x1 + 5, y1 - 6),
        font,
        font_scale,
        (255, 255, 255),
        thickness
    )


def run_filtered_model(
    model,
    image_array,
    confidence,
    allowed_keywords,
    blocked_keywords,
    model_label,
    color=(255, 0, 0),
):
    """
    Runs YOLO, filters unwanted classes, and draws only accepted detections.

    Important:
    This prevents Belt Surface Damage mode from showing Human / Roller / Other Objects.
    """

    result = model(image_array, conf=confidence, verbose=False)[0]

    annotated = image_array.copy()
    overlay = annotated.copy()
    detections = []

    names = result.names
    boxes = result.boxes
    masks_xy = result.masks.xy if result.masks is not None else None

    if boxes is None:
        return result, annotated, detections

    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if isinstance(names, dict):
            class_name = names.get(cls_id, str(cls_id))
        else:
            class_name = names[cls_id]

        # Filter unwanted classes
        if not class_is_allowed(class_name, allowed_keywords, blocked_keywords):
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()

        # Draw segmentation mask if available
        if masks_xy is not None and i < len(masks_xy):
            pts = masks_xy[i].astype(np.int32)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(annotated, [pts], True, color, 2)

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Better label format: Tear 25% instead of Tear 0.25
        label = f"{class_name.title()} {conf * 100:.0f}%"
        draw_label(annotated, label, x1, y1, color)

        detections.append(
            {
                "model": model_label,
                "class": class_name,
                "confidence": conf,
                "confidence_percent": round(conf * 100, 1),
                "box": [x1, y1, x2, y2],
            }
        )

    # Transparent mask overlay
    annotated = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)

    return result, annotated, detections


def get_raw_detections(result):
    try:
        if result is None or result.boxes is None:
            return []

        raw = []
        names = result.names

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if isinstance(names, dict):
                class_name = names.get(cls_id, str(cls_id))
            else:
                class_name = names[cls_id]

            raw.append({
                "class": class_name,
                "confidence": round(conf, 3),
                "confidence_percent": round(conf * 100, 1)
            })

        return raw

    except Exception as e:
        return [{"debug_error": str(e)}]

def combine_images(base_img, overlay_img):
    """
    Placeholder helper.
    Currently returns the latest annotated image.
    """
    return overlay_img


def recommended_confidence(mode):
    """
    Different models need different thresholds.

    Belt damage is kept sensitive because early tear/burn/patch warnings may appear
    at lower confidence.
    """

    if mode == "Belt Surface Damage":
        return 0.10

    if mode == "Spillage / Foreign Object":
        return 0.55

    if mode == "Idler / Roller Mechanical Health":
        return 0.60

    return 0.45

# =========================================================================
# --- TABS ---
# =========================================================================

tab1, tab2, tab3 = st.tabs(
    ["🚨 AI Vision Inspection", "📝 Manual Override (Codes)", "🛠️ Maintenance Scheduler"]
)


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
            unsafe_allow_html=True,
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
                unsafe_allow_html=True,
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
                unsafe_allow_html=True,
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
            unsafe_allow_html=True,
        )

    st.markdown("### 🧠 Select Inspection View")

    inspection_mode = st.selectbox(
        "Choose the type of image you are uploading:",
        [
            "Belt Surface Damage",
            "Spillage / Foreign Object",
            "Idler / Roller Mechanical Health",
            "Full Multi-Agent Scan",
        ],
    )

    if inspection_mode == "Belt Surface Damage":
        st.warning(
            "Use Belt Surface Damage only when belt rubber is visible. If coal/rocks cover the belt, use Spillage / Foreign Object mode."
        )

    st.caption(
        "False detections are reduced by running only the correct specialist model and filtering unwanted classes."
    )

    st.markdown("### 🎛️ AI Calibration")
    default_conf = recommended_confidence(inspection_mode)

    admin_password = st.text_input("Enter Admin Password to Unlock Calibration:", type="password")

    if admin_password == "dgms2026":
        st.success("🔓 Calibration Unlocked")
        confidence_threshold = st.slider(
            "Detection Confidence Threshold",
            0.05,
            0.99,
            default_conf,
            0.01,
        )
    else:
        st.info(f"🔒 System running at recommended threshold ({default_conf:.2f}).")
        confidence_threshold = default_conf

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Drag and drop or click to upload", type=["jpg", "png", "jpeg"]
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

                conveyor_dets = []
                spillage_dets = []
                idler_dets = []

                annotated_img = img_array.copy()

                if inspection_mode == "Belt Surface Damage":
                    res_conveyor, annotated_img, conveyor_dets = run_filtered_model(
                        conveyor_agent,
                        img_array,
                        confidence_threshold,
                        BELT_ALLOWED,
                        BELT_BLOCKED,
                        "Belt Surface Damage",
                        color=(255, 70, 70),
                    )

                elif inspection_mode == "Spillage / Foreign Object":
                    res_spillage, annotated_img, spillage_dets = run_filtered_model(
                        spillage_agent,
                        img_array,
                        confidence_threshold,
                        SPILLAGE_ALLOWED,
                        SPILLAGE_BLOCKED,
                        "Spillage / Foreign Object",
                        color=(255, 165, 0),
                    )

                elif inspection_mode == "Idler / Roller Mechanical Health":
                    res_idler, annotated_img, idler_dets = run_filtered_model(
                        idler_agent,
                        img_array,
                        confidence_threshold,
                        IDLER_ALLOWED,
                        IDLER_BLOCKED,
                        "Idler / Roller Mechanical Health",
                        color=(80, 180, 255),
                    )

                else:
                    res_conveyor, annotated_img, conveyor_dets = run_filtered_model(
                        conveyor_agent,
                        annotated_img,
                        max(confidence_threshold, 0.20),
                        BELT_ALLOWED,
                        BELT_BLOCKED,
                        "Belt Surface Damage",
                        color=(255, 70, 70),
                    )
                    res_spillage, annotated_img, spillage_dets = run_filtered_model(
                        spillage_agent,
                        annotated_img,
                        max(confidence_threshold, 0.55),
                        SPILLAGE_ALLOWED,
                        SPILLAGE_BLOCKED,
                        "Spillage / Foreign Object",
                        color=(255, 165, 0),
                    )
                    res_idler, annotated_img, idler_dets = run_filtered_model(
                        idler_agent,
                        annotated_img,
                        max(confidence_threshold, 0.60),
                        IDLER_ALLOWED,
                        IDLER_BLOCKED,
                        "Idler / Roller Mechanical Health",
                        color=(80, 180, 255),
                    )

            st.image(annotated_img, use_container_width=True)

        with col_results:
            st.markdown("### Inspection Verdict:")

            all_dets = conveyor_dets + spillage_dets + idler_dets
            total_anomalies = len(all_dets)

            with st.expander("🔍 Debug: Model Raw Output"):
                st.write("Inspection mode:", inspection_mode)
                st.write("Confidence threshold:", confidence_threshold)

                if res_conveyor is not None:
                    st.write("Conveyor model classes:", res_conveyor.names)
                    st.write(
                        "Raw conveyor detections before filtering:",
                        get_raw_detections(res_conveyor)
                    )
                    st.write(
                        "Accepted conveyor detections after filtering:",
                        conveyor_dets
                    )

                if res_spillage is not None:
                    st.write("Spillage model classes:", res_spillage.names)
                    st.write(
                        "Raw spillage detections before filtering:",
                        get_raw_detections(res_spillage)
                    )
                    st.write(
                        "Accepted spillage detections after filtering:",
                        spillage_dets
                    )

                if res_idler is not None:
                    st.write("Idler model classes:", res_idler.names)
                    st.write(
                        "Raw idler detections before filtering:",
                        get_raw_detections(res_idler)
                    )
                    st.write(
                        "Accepted idler detections after filtering:",
                        idler_dets
                    )

            if total_anomalies > 0:
                st.error(f"🚨 {total_anomalies} ANOMALIES DETECTED")
                st.error("Action: Dispatch maintenance team to verify zones.")

                for det in all_dets:
                    if det["model"] == "Belt Surface Damage":
                    st.warning(
                        f"Belt Surface Damage detected with {det['confidence'] * 100:.1f}% confidence."
                    )
                    else:
                        st.warning(
                            f"{det['model']}: {det['class']} detected with {det['confidence'] * 100:.1f}% confidence."
                        )

                st.info(
                    "📋 DGMS Recommendation:\n"
                    "- Immediate physical inspection required.\n"
                    "- Log incident in statutory register.\n"
                    "- Stop conveyor if defect is severe."
                )

            else:
                st.success("✅ NO AI-CONFIRMED ANOMALY")
                st.info("If visible damage is present, mark this image for retraining/manual inspection.")
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
    st.write("Use your device microphone/dictation. English, Hindi, and Hinglish are supported.")

    lang = st.radio("Response Language / उत्तर की भाषा:", ["English", "हिंदी"], horizontal=True)

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

        if st.button("📥 Generate Statutory PDF" if lang == "English" else "📥 वैधानिक PDF जनरेट करें"):
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
                type="primary",
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
        delta_color="inverse",
    )
    col2.metric(
        label="Idler Greasing Status",
        value="Overdue",
        delta="Action Req",
        delta_color="inverse",
    )
    col3.metric(
        label="Belt Tension & Alignment",
        value="Normal",
        delta="14 Days left",
        delta_color="normal",
    )

    st.markdown("---")
    st.markdown("#### 📋 DGMS Compliance Tracker")
    st.checkbox("Weekly Belt Inspection", value=True, disabled=True)
    st.checkbox("Emergency Pull Cord Test", value=True, disabled=True)
    st.checkbox("Fire Extinguisher & Sprinkler Check", value=False, disabled=True)
    st.checkbox("Walkthrough Record Updated", value=True, disabled=True)
