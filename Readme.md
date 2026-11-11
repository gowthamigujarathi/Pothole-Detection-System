#  Pothole Detection using YOLOv8

This repository contains a **trained YOLOv8 object detection model** along with a **simple inference script** to detect potholes in road images.

The project is designed to demonstrate **end-to-end pothole detection**, from a trained deep learning model to practical image-based inference. It serves as a strong foundation for future extensions such as **mobile camera detection, web deployment, GPS-based mapping, and smart road monitoring systems**.

---

##  Project Overview

- Detect potholes in road images using **YOLOv8**
- Run offline inference on a single image
- Visualize detections with bounding boxes and confidence scores
- Lightweight and easy to test
- Suitable for research, learning, and future deployment

---

##  Model Overview

- **Model Architecture:** YOLOv8  
- **Task:** Object Detection  
- **Detected Class:** Pothole  
- **Framework:** Ultralytics YOLO  
- **Input:** Road images  
- **Output:** Bounding boxes with confidence scores  
- **Model File:** `best.pt`  

---

##  Training Details

- **Base Model:** YOLOv8 (pretrained on COCO)
- **Training Type:** Transfer Learning
- **Dataset Format:** YOLO (images + labels)
- **Number of Classes:** 1 (`pothole`)
- **Training Environment:** Google Colab (GPU)
- **Training Strategy:**
  - Data augmentation enabled
  - Mosaic and MixUp augmentation
  - HSV color augmentation
  - Horizontal flipping
  - Scaling and translation
  - Trained for multiple epochs to ensure convergence

---

##  Model Performance

The model was evaluated on a validation dataset with the following results:

| Metric        | Value |
|--------------|-------|
| Precision     | **0.91** |
| Recall        | **0.775** |
| mAP@50        | **0.859** |
| mAP@50–95     | **0.505** |

### Performance Interpretation
- **High Precision:** Very few false positives
- **Good Recall:** Most potholes are detected
- **Strong mAP@50:** Accurate bounding box localization
- **Reliable generalization** on real-world road images

---

##  Inference Script

The repository includes a **simple Python inference script** that:
- Loads the trained YOLOv8 model
- Accepts an input image
- Runs pothole detection
- Draws bounding boxes with confidence scores
- Saves the output image

This script is intended for **testing and evaluating the trained model**.

##  Project Structure
pothole-detection/
│
├── predict.py # Inference script
├── best.pt # Trained YOLOv8 model  
├── input.jpg # Input image
├── output.jpg # Output image (generated)
├── requirements.txt # Python dependencies
└── README.md # Project documentation

##  Installation Guide

### Clone the repository

git clone https://github.com/your-username/pothole-detection.git
cd pothole-detection

### Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows

### Install required dependencies
bash
Copy code
pip install -r requirements.txt

###  How to Run the Project
1. Place your trained YOLOv8 model in the project directory:
best.pt

2. Add the image you want to test:
input.jpg

3. Run the inference script:
python predict.py

4. The output image will be saved as:
output.jpg

### Output
The system detects potholes and marks them using green bounding boxes along with confidence scores, allowing easy visual evaluation of the model's performance.

### Author
Aryan Kuamr Yadav
projects.aky@gmail.com