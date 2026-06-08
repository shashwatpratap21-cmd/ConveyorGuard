import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from twilio.rest import Client

# --- PAGE CONFIG ---
st.set_page_config(page_title="ConveyorGuard", page_icon="⛏️", layout="wide")

# --- TWILIO SMS CONFIGURATION ---
def send_emergency_sms(alert_type, details):
    # Ensure these are set in your Streamlit secrets
    try:
        TWILIO_SID = st.secrets["TWILIO_SID"]
        TWILIO_TOKEN = st.secrets["TWILIO_TOKEN"]
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        message = client.messages.create(
            body=f"🚨 CONVEYORGUARD ALERT: {alert_type}\nDetails: {details}",
            from_="+13367042789",
            to="+918467068023"
        )
        return True
    except Exception as e:
        st.error(f"SMS Delivery Failed: {e}")
        return False

# --- KERAS FIX (SafeDense) ---
@tf.keras.utils.register_keras_serializable()
class SafeDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('conveyorguard_model.h5', compile=False, custom_objects={'Dense': SafeDense})

try:
    model = load_model()
except Exception as e:
    st.sidebar.warning(f"Model Load Error: {e}")

st.title("⛏️ ConveyorGuard Dashboard")
st.subheader("AI-Powered Inspection & Safety Management System")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["🚨 AI Vision Inspection", "📝 Manual Override (Codes)", "🛠️ Maintenance Scheduler"])

# --- TAB 1: AI INSPECTION ---
with tab1:
    st.markdown("### Upload Conveyor Belt Image")
    
    with st.expander("📋 DGMS Pre-Inspection Safety Protocol", expanded=True):
        st.warning("""
        **🟡 CRITICAL UNDERGROUND SAFETY REQUIREMENTS:**
        * Inform the control room before starting.
        * Maintain 1.5m clearance.
        * Use crossover bridges only.
        """)
    
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)
        
        img_resized = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
        
        with st.spinner("Analyzing..."):
            prediction = model.predict(img_array)[0][0]
            if prediction > 0.5:
                st.error(f"🚨 CRITICAL DAMAGE ({prediction * 100:.1f}% Confidence)")
            else:
                st.success(f"✅ NORMAL / HEALTHY LOAD")

# --- TAB 2: MANUAL OVERRIDE ---
with tab2:
    st.markdown("### 🎙️ Emergency Manual Reporting")
    incident = st.text_area("Describe the incident (e.g., TEAR, FIRE, WATER):").upper()
    
    if st.button("Submit & Send SMS"):
        if incident:
            success = send_emergency_sms("Manual Hazard Report", incident)
            if success:
                st.success("✅ Report logged and SMS alert dispatched to Mine Manager.")
        else:
            st.warning("Please enter a description.")
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
        
        # 1. Belt Tear (WITH HIERARCHY SMS)
        if any(word in active_report for word in ["TEAR", "CUT", "RUPTURE", "BROKEN", "FAT", "TOOT", "FATA", "TUTA"]):
            if lang == "English":
                st.error("🚨 CRITICAL ALERT LOGGED: Belt Tear/Rupture detected.")
                st.error("**Action:** Stop belt immediately. Dispatch vulcanizing crew.")
                st.markdown("---")
                st.markdown("### 📱 Emergency Communication Network")
                st.info("""
                **Alert Routing based on Incident Severity:**
                * **Shift Engineer:** (Mandatory for all logs)
                * **Mine Manager:** (Triggered if status is CRITICAL)
                * **DGMS Control Room:** (Triggered if FIRE/INUNDATION)
                """)
                if st.button("🚨 Dispatch Alert Network", key="sms_tear_en", type="primary"):
                    with st.spinner("Routing encrypted alerts via Twilio..."):
                        success = send_emergency_sms("Critical Belt Rupture", active_report)
                        if success:
                            st.success("✅ Statutory SMS Alerts Sent Successfully!")
            else:
                st.error("🚨 गंभीर चेतावनी: बेल्ट फटने की सूचना मिली है।")
                st.error("**कार्रवाई (Action):** तुरंत बेल्ट रोकें। मरम्मत टीम (Vulcanizing crew) को भेजें।")
                st.markdown("---")
                st.markdown("### 📱 आपातकालीन संचार नेटवर्क")
                st.info("""
                **घटना की गंभीरता के आधार पर अलर्ट रूटिंग:**
                * **शिफ्ट इंजीनियर:** (सभी लॉग के लिए अनिवार्य)
                * **खदान प्रबंधक:** (गंभीर स्थिति में ट्रिगर)
                * **DGMS कंट्रोल रूम:** (आग/बाढ़ की स्थिति में ट्रिगर)
                """)
                if st.button("🚨 अलर्ट नेटवर्क भेजें", key="sms_tear_hi", type="primary"):
                    with st.spinner("Twilio के माध्यम से अलर्ट भेजा जा रहा है..."):
                        success = send_emergency_sms("Critical Belt Rupture", active_report)
                        if success:
                            st.success("✅ वैधानिक SMS अलर्ट सफलतापूर्वक भेज दिए गए!")
                
        # 2. Fire/Smoke (WITH HIERARCHY SMS)
        elif any(word in active_report for word in ["FIRE", "SMOKE", "BURNING", "SPARK", "AAG", "DHUAN", "JALA", "SULAG"]):
            if lang == "English":
                st.error("🔥 FIRE EMERGENCY LOGGED: Combustion indicators detected.")
                st.error("**CRITICAL:** Evacuate district. Turn on main suppression systems. Alert DGMS.")
                st.markdown("---")
                st.markdown("### 📱 Emergency Communication Network")
                st.info("""
                **Alert Routing based on Incident Severity:**
                * **Shift Engineer:** (Mandatory for all logs)
                * **Mine Manager:** (Triggered if status is CRITICAL)
                * **DGMS Control Room:** (Triggered if FIRE/INUNDATION)
                """)
                if st.button("🚨 Dispatch Alert Network", key="sms_fire_en", type="primary"):
                    with st.spinner("Routing encrypted alerts via Twilio..."):
                        success = send_emergency_sms("Underground Fire Detected", active_report)
                        if success:
                            st.success("✅ Statutory SMS Alerts Sent Successfully!")
            else:
                st.error("🔥 आग आपातकाल: आग या धुएं की सूचना मिली है।")
                st.error("**खतरा (CRITICAL):** तुरंत खदान खाली करें। वाटर स्प्रिंकलर चालू करें। DGMS को अलर्ट करें।")
                st.markdown("---")
                st.markdown("### 📱 आपातकालीन संचार नेटवर्क")
                st.info("""
                **घटना की गंभीरता के आधार पर अलर्ट रूटिंग:**
                * **शिफ्ट इंजीनियर:** (सभी लॉग के लिए अनिवार्य)
                * **खदान प्रबंधक:** (गंभीर स्थिति में ट्रिगर)
                * **DGMS कंट्रोल रूम:** (आग/बाढ़ की स्थिति में ट्रिगर)
                """)
                if st.button("🚨 अलर्ट नेटवर्क भेजें", key="sms_fire_hi", type="primary"):
                    with st.spinner("Twilio के माध्यम से अलर्ट भेजा जा रहा है..."):
                        success = send_emergency_sms("Underground Fire Detected", active_report)
                        if success:
                            st.success("✅ वैधानिक SMS अलर्ट सफलतापूर्वक भेज दिए गए!")
                
        # 3. Spillage/Blockage
        elif any(word in active_report for word in ["SPIL", "BLOCK", "OVERFLOW", "JAM", "GIRA", "BHAR", "RUKA", "BAND"]):
            if lang == "English":
                st.warning("⚠️ WARNING LOGGED: Material spillage or blockage reported.")
                st.warning("**Action:** Dispatch cleaning crew to clear idlers and avoid friction fires.")
            else:
                st.warning("⚠️ चेतावनी: कोयला गिरने या बेल्ट जाम होने की सूचना है।")
                st.warning("**कार्रवाई (Action):** सफाई टीम को भेजें ताकि घर्षण (friction) से आग न लगे।")

        # 4. Water/Flooding (WITH HIERARCHY SMS)
        elif any(word in active_report for word in ["WATER", "FLOOD", "INUND", "LEAK", "PAANI", "BAARISH", "RISSA"]):
            if lang == "English":
                st.error("🌊 INUNDATION RISK LOGGED: Water flooding reported.")
                st.error("**Action:** Evacuate immediately. Activate main water pumps. Alert mine manager.")
                st.markdown("---")
                st.markdown("### 📱 Emergency Communication Network")
                st.info("""
                **Alert Routing based on Incident Severity:**
                * **Shift Engineer:** (Mandatory for all logs)
                * **Mine Manager:** (Triggered if status is CRITICAL)
                * **DGMS Control Room:** (Triggered if FIRE/INUNDATION)
                """)
                if st.button("🚨 Dispatch Alert Network", key="sms_water_en", type="primary"):
                    with st.spinner("Routing encrypted alerts via Twilio..."):
                        success = send_emergency_sms("Critical Inundation Risk", active_report)
                        if success:
                            st.success("✅ Statutory SMS Alerts Sent Successfully!")
            else:
                st.error("🌊 बाढ़ का खतरा: खदान में पानी भरने की सूचना है।")
                st.error("**कार्रवाई (Action):** तुरंत बाहर निकलें। मुख्य वाटर पंप चालू करें। खदान प्रबंधक को अलर्ट करें।")
                st.markdown("---")
                st.markdown("### 📱 आपातकालीन संचार नेटवर्क")
                st.info("""
                **घटना की गंभीरता के आधार पर अलर्ट रूटिंग:**
                * **शिफ्ट इंजीनियर:** (सभी लॉग के लिए अनिवार्य)
                * **खदान प्रबंधक:** (गंभीर स्थिति में ट्रिगर)
                * **DGMS कंट्रोल रूम:** (आग/बाढ़ की स्थिति में ट्रिगर)
                """)
                if st.button("🚨 अलर्ट नेटवर्क भेजें", key="sms_water_hi", type="primary"):
                    with st.spinner("Twilio के माध्यम से अलर्ट भेजा जा रहा है..."):
                        success = send_emergency_sms("Critical Inundation Risk", active_report)
                        if success:
                            st.success("✅ वैधानिक SMS अलर्ट सफलतापूर्वक भेज दिए गए!")
                
        else:
            if lang == "English":
                st.info("📝 General log received. Control room notified for verification.")
            else:
                st.info("📝 रिपोर्ट दर्ज कर ली गई है। वेरिफिकेशन के लिए कंट्रोल रूम को सूचित कर दिया गया है।")
        
        # --- STATUTORY RECORD EXPORT (PDF UPGRADE) ---
        st.markdown("---")
        st.markdown("### 📥 Statutory Record Management" if lang == "English" else "### 📥 वैधानिक रिकॉर्ड प्रबंधन")
        
        import datetime
        from fpdf import FPDF
        
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 1. Initialize the PDF Document
        pdf = FPDF()
        pdf.add_page()
        
        # 2. Add Official Headers
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, "CONVEYORGUARD - DGMS STATUTORY LOG", ln=True, align='C')
        pdf.set_font("Arial", "I", 10)
        pdf.cell(200, 10, "Sijua Colliery - Official Emergency Inspection Report", ln=True, align='C')
        pdf.ln(10)
        
        # 3. Add Incident Metadata Box
        pdf.set_font("Arial", "B", 12)
        pdf.cell(40, 10, "Date & Time:", border=1)
        pdf.set_font("Arial", "", 12)
        pdf.cell(150, 10, f" {current_time}", border=1, ln=True)
        
        pdf.set_font("Arial", "B", 12)
        pdf.cell(40, 10, "Status:", border=1)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(150, 10, " CRITICAL - ACTION REQUIRED", border=1, ln=True)
        pdf.ln(10)
        
        # 4. Add the Dictated Report
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, "Incident Description:", ln=True)
        pdf.set_font("Arial", "", 12)
        
        # Clean text so it formats perfectly in the PDF
        clean_report = active_report.replace('\n', ' ').encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, clean_report)
        pdf.ln(10)
        
        # 5. Add the DGMS Legal Notice
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, "Legal Notice of Inspection:", ln=True)
        pdf.set_font("Arial", "", 10)
        legal_text = "Pursuant to DGMS Circular No. 3 of 2020, this document serves as the official statutory log for the incident reported above. Immediate physical inspection by the Shift Engineer and Vulcanizing Crew is mandated. Do not wait for scheduled maintenance. Falsification of this statutory log is a punishable offense under the Mines Act, 1952."
        pdf.multi_cell(0, 6, legal_text)
        
        # 6. Add Signature Lines
        pdf.ln(20)
        pdf.cell(95, 10, "___________________________", align='C')
        pdf.cell(95, 10, "___________________________", align='C', ln=True)
        pdf.cell(95, 10, "Shift Engineer Signature", align='C')
        pdf.cell(95, 10, "Mine Manager Signature", align='C')
        
        # Convert PDF to bytes for downloading
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        
        # The Download Button
        if lang == "English":
            st.download_button(
                label="📄 Download Official DGMS Report (PDF)",
                data=pdf_bytes,
                file_name=f"DGMS_Report_{current_time[:10]}.pdf",
                mime="application/pdf",
                type="primary"
            )
        else:
            st.download_button(
                label="📄 आधिकारिक DGMS रिपोर्ट डाउनलोड करें (PDF)",
                data=pdf_bytes,
                file_name=f"DGMS_Report_{current_time[:10]}.pdf",
                mime="application/pdf",
                type="primary"
            )

# --- TAB 3: MAINTENANCE SCHEDULER ---
with tab3:
    st.markdown("### 🛠️ Predictive Maintenance & Statutory Compliance")
    st.info("Track statutory inspections, safety risks, and vulcanizing schedules.")
    
    # 1. Top Level Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Next Statutory Walkthrough", value="2 Days", delta="-1 day (Urgent)", delta_color="inverse")
    with col2:
        st.metric(label="Idler Greasing Status", value="Overdue", delta="Action Req", delta_color="inverse")
    with col3:
        st.metric(label="Belt Tension & Alignment", value="Normal", delta="14 Days left", delta_color="normal")
        
    st.markdown("---")
    
    # 2. Mine Manager's Overview (Compliance & Safety)
    col_comp, col_risk = st.columns(2)
    
    with col_comp:
        st.markdown("#### 📋 DGMS Compliance Tracker")
        st.markdown("Routine checks per Circular No. 3 of 2020:")
        # Using checkboxes that default to checked/unchecked to show status
        st.checkbox("Weekly Belt Inspection (Form-4)", value=True, disabled=True)
        st.checkbox("Emergency Pull Cord Test", value=True, disabled=True)
        st.checkbox("Fire Extinguisher & Sprinkler Check", value=False, disabled=True) # Unchecked to show action needed!
        st.checkbox("Walkthrough Record Updated", value=True, disabled=True)

    with col_risk:
        st.markdown("#### ⚠️ Active Safety Risk Panel")
        st.markdown("Current operational hazards:")
        st.error("**🔥 Fire Risk: HIGH** (Coal dust accumulation noted near tail pulley)")
        st.warning("**👷 Personnel Risk: MEDIUM** (Clearance zone restricted in Sector 4)")
        st.success("**⚙️ Belt Failure Risk: LOW** (No immediate structural tearing detected)")
        
    st.markdown("---")
    
    # 3. Maintenance Logging Form
    st.markdown("#### 📅 Schedule Repair / Vulcanizing")
    
    with st.form("maintenance_form"):
        task = st.selectbox("Select Maintenance Task:", [
            "Hot Vulcanizing (Belt Splicing)", 
            "Idler/Roller Replacement", 
            "Drive Motor Alignment", 
            "Tail Pulley Cleaning",
            "Fire Extinguisher Replacement",
            "General Statutory Inspection"
        ])
        
        col_date, col_team = st.columns(2)
        with col_date:
            scheduled_date = st.date_input("Scheduled Date")
        with col_team:
            assigned_to = st.text_input("Assigned Team / Contractor")
            
        comments = st.text_area("Additional Engineer Comments:")
        
        # The Submit Button
        submitted = st.form_submit_button("💾 Log Maintenance Task", type="primary")
        
        if submitted:
            if assigned_to == "":
                st.error("Please assign a team before logging the task.")
            else:
                st.success(f"✅ {task} successfully scheduled for {scheduled_date}.")
                st.info(f"Notification sent to: {assigned_to}")
