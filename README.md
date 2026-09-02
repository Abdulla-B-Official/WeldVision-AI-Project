# 🔧 WeldVision AI — Real-Time Weld Quality Inspection & Defect Detection

[![Live Demo](https://img.shields.io/badge/Render-Live%20Demo-46E3B7?style=for-the-badge&logo=render&logoColor=black)](https://weldvision-ai-project.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF9900?style=for-the-badge)](https://docs.ultralytics.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-CPU_Optimized-00599C?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Roboflow](https://img.shields.io/badge/Roboflow-Data_Prep-6706CE?style=for-the-badge&logo=roboflow&logoColor=white)](https://roboflow.com/)

<p align="center">
  <b>AI-powered industrial computer vision system for automatically identifying, classifying, and inspecting weld quality using a custom-trained, CPU-optimized YOLOv8s ONNX model.</b>
</p>

---

## 🔗 Live Application

🔗 **[weldvision-ai-project.onrender.com](https://weldvision-ai-project.onrender.com)**

<p align="center">
  <img src="https://img.shields.io/badge/Model-YOLOv8s_ONNX-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Classes-3-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Test_mAP%4050-79.98%25-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Test_Precision-77.94%25-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Test_Recall-73.71%25-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Test_mAP%4050--95-50.29%25-purple?style=for-the-badge" />
</p>

---

## Overview

**WeldVision AI** is an AI-powered industrial computer vision system designed to automatically identify, classify, and inspect weld quality using a custom-trained, CPU-optimized **YOLOv8s ONNX object detection model**.

The system focuses on three specific weld classifications:

* 🟢 **Good Weld** (Acceptable)
* 🔴 **Bad Weld** (Defective)
* ⚠️ **Defect** (Anomalies / Porosity / Cracks)

Instead of relying on subjective manual inspection, the system uses deep learning to locate weld beads, draw bounding boxes around critical zones, classify the quality, and provide a confidence score for every detection. Furthermore, it supports **batch processing** for high-volume industrial inspection.

The optimized model is integrated into a lightweight **Flask API** and deployed as a live web application using **Render**, allowing quality assurance teams to perform single or batch detections directly from a browser.

---

### Application Features

* Single and batch image upload support
* Automatic weld defect detection & classification
* Bounding-box visualization
* Confidence scores & dynamic threshold control
* Live webcam inspection for real-time QA
* High-volume batch reporting
* API/model status monitoring

---

## Project Objective

The primary objective is to build an automated industrial weld inspection system that can:

* Automatically detect weld joints and defects from images and video feeds
* Classify objects into **Good Weld**, **Bad Weld**, and **Defect**
* Process multiple images concurrently using efficient batch inference
* Locate specific defects using precise bounding boxes
* Provide confidence scores for quality assurance tracking
* Reduce inspector fatigue and dependence on manual sampling
* Execute efficiently on CPU-only cloud instances using ONNX Runtime
* Deploy the trained AI model as a scalable online REST service

---

## Problem Statement

Manual visual inspection of industrial welds is critical but highly prone to inspector fatigue, subjectivity, and human error. Undetected weld defects can lead to catastrophic structural failures.

Traditional weld inspection pipelines often rely on manual identification, which is:

* Time-consuming and inefficient for high-volume manufacturing
* Highly subjective between different human inspectors
* Difficult to scale across multiple production lines
* Lacking in automated, digitized quality logging

### Proposed Solution

This project introduces an AI-powered, edge-optimized object detection pipeline that automatically:

$$\text{Ingest (Single/Batch)} \longrightarrow \text{Detect} \longrightarrow \text{Classify} \longrightarrow \text{Localize} \longrightarrow \text{Report}$$

For every detection, the system produces:

```text
Weld Quality Class
Bounding Box Coordinates
Confidence Score
Aggregated Batch Quality Report
