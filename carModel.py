# Import necessary libraries
from ultralytics import YOLO
import cv2
import time

# Step 1: Define model paths
model_paths = {
    "best": "runs/train/car_detection_fast2/weights/best.pt"  # Best performing model
      # Last epoch model
}

# Step 2: Function to test a model with webcam
def test_model_with_webcam(model_path, model_name):
    print(f"Testing {model_name} model...")
    model = YOLO(model_path)  # Load the specified model

    # Initialize the webcam
    cap = cv2.VideoCapture(0)  # 0 is the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set up webcam parameters (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Process live feed
    print(f"Press 'q' to quit the {model_name} webcam feed. Press 's' to switch to the next model.")
    start_time = time.time()
    frame_count = 0

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Run YOLOv8 inference on the frame
        results = model.predict(frame, conf=0.5)  # Confidence threshold 0.5

        # Plot the results on the frame
        annotated_frame = results[0].plot()  # Draws bounding boxes and labels

        # Calculate FPS
        frame_count += 1
        if frame_count % 10 == 0:  # Update FPS every 10 frames
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            print(f"{model_name} FPS: {fps:.2f}")
            start_time = end_time
            frame_count = 0

        # Display the annotated frame
        cv2.imshow(f"YOLOv8 {model_name} Webcam Detection", annotated_frame)

        # Break or switch based on key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and len(model_paths) > 1:
            cap.release()
            cv2.destroyAllWindows()
            remaining_models = [m for m in model_paths.keys() if m != model_name]
            if remaining_models:
                next_model = remaining_models[0]
                test_model_with_webcam(model_paths[next_model], next_model)
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print(f"{model_name} webcam feed stopped.")

# Step 3: Test both models
for model_name, model_path in model_paths.items():
    test_model_with_webcam(model_path, model_name)