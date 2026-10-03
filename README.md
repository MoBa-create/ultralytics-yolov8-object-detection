# ⚡ YOLOv8 Object Detection Pipeline

An efficient and lightweight Object Detection project built with **Ultralytics YOLOv8** (Nano version) in PyTorch. This pipeline detects up to 80 COCO classes from both local images and online URLs with real-time inference speed.

---

## 📌 Features

- **Single-Stage Architecture:** Fast inference using standard YOLOv8 pre-trained weights.
- **Flexible Inputs:** Accepts both local image files and direct web image URLs.
- **CLI Interface:** Supports custom execution via command-line arguments (confidence threshold, custom inputs, custom outputs).
- **Automated Visualization:** Draws high-contrast bounding boxes with class names and confidence percentages.

---

## 📁 Project Structure

```text
.
├── detect_yolo.py     # Main YOLOv8 inference script
├── requirements.txt   # Dependencies
├── .gitignore         # Git ignore file
└── README.md          # Project documentation