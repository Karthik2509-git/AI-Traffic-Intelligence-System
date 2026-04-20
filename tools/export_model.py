from ultralytics import YOLO
import os

# Create data directory if missing
if not os.path.exists('data'):
    os.makedirs('data')

print("[ATOS] Exporting YOLOv8 model for 4K Optimization...")
model = YOLO('yolov8m.pt') # Upgrading to Medium for better accuracy

# Export to ONNX
model.export(format='onnx', imgsz=960, opset=12)

# Move to the project data directory
onnx_file = 'yolov8m.onnx'
if os.path.exists(onnx_file):
    os.replace(onnx_file, 'data/yolov8_4k_optimized.onnx')
    print("[ATOS] Success: Model exported to data/yolov8_4k_optimized.onnx")
else:
    print("[ERROR] Export failed: " + onnx_file + " not found.")
