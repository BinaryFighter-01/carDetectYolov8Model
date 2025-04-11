Below is a sample `README.md` file tailored for your GitHub repository. It provides an overview of your project, instructions for setting it up, and specifies running `carModel.py` for object detection. Feel free to customize it with more details about your dataset, model, or additional features.

---

# Car Detection with YOLOv8

This repository contains a project for detecting cars in images and live webcam feeds using the YOLOv8 object detection model. The model is trained on a custom dataset to identify cars with high accuracy and optimized for fast performance.

## Overview
- **Model**: YOLOv8 (nano variant, `yolov8n`, for speed).
- **Task**: Object detection of cars.
- **Dataset**: A custom dataset prepared using Roboflow, containing images of cars.
- **Hardware**: Tested on a CPU (12th Gen Intel Core i7-1255U); GPU support optional.
- **Purpose**: Real-time car detection for applications like surveillance or autonomous systems.

## Features
- Trains a YOLOv8 model on a car dataset.
- Evaluates model performance with metrics like mAP, precision, and FPS.
- Supports live webcam detection for real-time testing.
- Optimized for fast training and inference.

## Prerequisites
- **Python 3.8+**
- **Required Libraries**:
  - `ultralytics` (for YOLOv8)
  - `opencv-python` (for webcam and image processing)
  - `torch` (PyTorch backend)
- **Installation**:
  ```bash
  pip install ultralytics opencv-python torch
  ```
- **Hardware**: A working webcam is recommended for live testing.

## Project Structure
```
car.v2i.yolov8/
├── dataset/
│   ├── data.yaml
│   ├── train/
│   │   ├── images/
│   │   ├── labels/
│   ├── valid/
│   │   ├── images/
│   │   ├── labels/
│   ├── test/
│       ├── images/
│       ├── labels/
├── carModel.py      # Main script for training and testing
├── README.md        # This file
└── runs/            # Directory for training outputs (auto-generated)
```

## Usage

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/car.v2i.yolov8.git
cd car.v2i.yolov8
```

### 2. Install Dependencies
Run the following command to install required libraries:
```bash
pip install ultralytics opencv-python torch
```

### 3. Run the Model
To train the model and perform object detection, execute the following command:
```bash
python carModel.py
```
- The script will:
  - Train the YOLOv8n model on the dataset for 10 epochs (optimized for speed).
  - Validate the model and display metrics (mAP@0.5, precision, recall, FPS).
  - Allow testing on the test set and display a sample result.
  - Optionally, run live webcam detection (press 'q' to quit).

### 4. Live Webcam Testing
After training, the script automatically switches to live webcam mode to detect cars in real-time. Use the following controls:
- Press `q` to quit the webcam feed.
- Press `s` to switch between `best.pt` and `last.pt` models (if available).

### 5. Output
- Training results and model weights are saved in `runs/train/car_detection_fast/`.
- Test set predictions are saved in `runs/detect/predict/`.
- The best model (`best.pt`) is used for inference.

## Configuration
- **Dataset Path**: Update `data` in `carModel.py` to match your dataset location if different from `C:/Users/Anil Abhange/Downloads/car.v2i.yolov8/dataset/data.yaml`.
- **Model**: Uses `yolov8n.pt` for fast performance; switch to `yolov8s.pt` for better accuracy if needed.
- **Hyperparameters**: Adjust `epochs`, `imgsz`, and `batch` in `carModel.py` for your needs.

## Troubleshooting
- **Webcam Issues**: Ensure your webcam is connected and try `cv2.VideoCapture(1)` if `0` fails.
- **Missing Files**: Verify the dataset structure and paths in `data.yaml`.
- **Performance**: Reduce `imgsz` (e.g., to 320) or `batch` (e.g., to 8) if memory is limited.

## Contributing
Feel free to fork this repository, submit issues, or propose enhancements. Contributions are welcome!
