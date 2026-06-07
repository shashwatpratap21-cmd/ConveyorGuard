# ⛏️ ConveyorGuard: AI Belt Inspection System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red)

**ConveyorGuard** is a computer vision-based safety application designed specifically for the **Sijua Colliery**. It utilizes a deep learning model to perform instant anomaly detection on underground conveyor belts, preventing catastrophic equipment failure and ensuring operational safety.

## 🚀 Live Demo
(https://conveyorguard-amwpdvzporhftv3ah78fya.streamlit.app/)

## 🧠 The Engineering Challenge
Moving a neural network from a research lab (Google Colab) to a live production server often introduces severe serialization and version-mismatch bugs. This application features custom **Keras Interceptor Layers** engineered specifically to bypass strict server-side security protocols. 

By writing custom classes to intercept native Keras operations (`TrueDivide`, `Subtract`, and `Dense` quantization configurations), this application successfully bridges the gap between training environments and strict cloud deployment constraints without retraining the core model.

## ⚙️ Core Features
* **Real-Time Analysis:** Instant processing of high-resolution belt imagery.
* **Deep Learning Architecture:** Powered by a customized MobileNetV2 base, optimized for edge-case industrial anomalies (tears, fraying, misalignment).
* **Actionable Intelligence:** Outputs strict safety verdicts, percentage-based confidence metrics, and step-by-step emergency protocols when anomalies are flagged.
* **Industrial UI:** Clean, responsive interface built for shift engineers on the ground.

## 🛠️ Technology Stack
* **Frontend/Hosting:** Streamlit (Cloud Deployment)
* **AI/Machine Learning:** TensorFlow, Keras
* **Data Processing:** NumPy, Pillow (PIL)

## 💻 Local Installation

To run this inspection system on your local machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/shashwatpratap21-cmd/ConveyorGuard.git](https://github.com/shashwatpratap21-cmd/ConveyorGuard.git)
   cd ConveyorGuard
