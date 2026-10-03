# ⚡ YOLOv8 Object Detection Pipeline

An efficient and lightweight Object Detection project built with **Ultralytics YOLOv8** (Nano version) in PyTorch. This pipeline detects up to 80 COCO classes from both local images and online URLs with real-time inference speed.

---

## 📌 Features

- **Single-Stage Architecture:** Fast inference using standard YOLOv8 pre-trained weights.
- **Flexible Inputs:** Accepts both local image files and direct web image URLs.
- **CLI Interface:** Supports custom execution via command-line arguments (confidence threshold, custom inputs, custom outputs).
- **Automated Visualization:** Draws high-contrast bounding boxes with class names and confidence percentages.

---

<img width="810" height="1080" alt="bus" src="https://github.com/user-attachments/assets/1a1e8f04-02d4-4e8d-9192-1e5bd898b693" />

<img width="810" height="1080" alt="yolo_output" src="https://github.com/user-attachments/assets/e28caed5-782c-4c72-839d-457a718c0581" />


## 📁 Project Structure

```text
.
├── detect_yolo.py     # Main YOLOv8 inference script
├── requirements.txt   # Dependencies
├── .gitignore         # Git ignore file
└── README.md          # Project documentation
