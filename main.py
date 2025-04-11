# Import necessary libraries
from ultralytics import YOLO
import os
import time
import shutil  # For cleaning up previous runs

# Step 1: Clean up previous runs (optional, to avoid confusion)
run_dir = "runs/train/car_detection"
if os.path.exists(run_dir):
    shutil.rmtree(run_dir)  # Deletes the entire car_detection folder
    print(f"Cleared previous run at {run_dir}")

# Step 2: Load the pre-trained YOLOv8n model
model = YOLO("yolov8n.pt")  # Switch to yolov8n.pt for faster performance

# Step 3: Train the model on your dataset with speed optimizations
print("Starting training in fast mode...")
model.train(
    data="C:/Users/Anil Abhange/Downloads/car.v2i.yolov8/dataset/data.yaml",  # Absolute path
    epochs=7,  # Increase to 10 for better convergence (adjust as needed)
    imgsz=1000,  # Reduce image size for faster training (default was 640)
    batch=8,   # Keep batch size; reduce to 8 if memory issues
    project="runs/train",
    name="car_detection_fast",
    exist_ok=False,  # Prevent overwriting without clearing
    save_period=-1,  # Save only best.pt and last.pt
    patience=5,      # Stop early if mAP doesn't improve for 5 epochs
    workers=4,       # Limit workers for CPU efficiency
    device="cpu"     # Explicitly use CPU (optional, defaults to available device)
)

# Step 4: Validate the model on the validation set
print("Validating the model...")
metrics = model.val()
print(f"mAP@0.5: {metrics.box.map50:.2f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.2f}")
print(f"Precision: {metrics.box.precision[0]:.2f}")
print(f"Recall: {metrics.box.recall[0]:.2f}")

# Step 5: Test inference on the test set
print("Running inference on test set...")
test_images_dir = "C:/Users/Anil Abhange/Downloads/car.v2i.yolov8/dataset/test/images"  # Absolute path
test_images = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
start_time = time.time()
results = model.predict(source=test_images_dir, save=True, conf=0.5)
end_time = time.time()

# Calculate FPS
fps = len(test_images) / (end_time - start_time)
print(f"FPS: {fps:.2f}")

# Step 6: Display a sample result (optional)
results[0].show()