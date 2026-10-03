import argparse
from ultralytics import YOLO
from PIL import Image

def run_yolo_detection(source, confidence=0.25, save_path="yolo_output.jpg"):
    model = YOLO("yolov8n.pt")

    print(f"⏳ Running YOLOv8 object detection on: {source}")
    results = model(source, conf=confidence)

    for result in results:
        boxes = result.boxes
        print(f"\n🎯 Total objects detected: {len(boxes)}")
        
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0]) * 100
            coords = box.xyxy[0].tolist()
            print(f"📍 Class: {class_name:<12} | Conf: {conf:.1f}% | Box: {[round(c, 1) for c in coords]}")

        annotated_array = result.plot()
        output_image = Image.fromarray(annotated_array[..., ::-1])
        output_image.save(save_path)
        print(f"\n✅ Result saved successfully to: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection Pipeline")
    parser.add_argument("--source", type=str, default="https://ultralytics.com/images/bus.jpg", help="Local image path or URL")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0.0 - 1.0)")
    parser.add_argument("--output", type=str, default="yolo_output.jpg", help="Path to save output image")

    args = parser.parse_args()
    run_yolo_detection(args.source, args.conf, args.output)