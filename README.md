# 🏏 Real-Time Cricket Player & Equipment Detection

**Project:** Cricket Analytics System  
**Technologies:** Python, YOLOv8 (Ultralytics), OpenCV, PyTorch, NumPy  

---

## 🔹 Overview

This project implements a **real-time computer vision system** that detects cricket players, classifies their roles (Batsman, Bowler, Wicket-Keeper), and identifies cricket equipment (bat, ball) from live video streams. It also supports object tracking, cropping, and live analytics for enhanced game insights.

---

## 🔹 Features

- **Player Classification:** Identifies cricket player roles in real-time.  
- **Equipment Detection:** Detects bats and balls accurately.  
- **Live Video Analytics:** Annotates detections on webcam/video input.  
- **Object Cropping & Tracking:** Crops detected objects automatically for further analysis.  
- **Visualization:** Displays live detection results and analytics overlays.  

---

## 🔹 Technical Details

- **Model:** Custom-trained YOLOv8 object detection model.  
- **Frameworks:** Python, OpenCV, PyTorch, Ultralytics YOLOv8.  
- **Performance:** Real-time detection on CPU with live annotation and tracking.  
- **Pipeline:** Video capture → Detection → Classification → Cropping → Analytics visualization.  

---

## 🔹 Key Achievements

- Built a **fully functional real-time cricket detection system**.  
- Implemented **object tracking and player role classification**.  
- Demonstrated **practical application of computer vision in sports analytics**.  

---

## 🔹 Demo / Usage

```bash
# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run webcam detection
python webcam_detect.py

# Run crop analytics
python crop_analytics.py
