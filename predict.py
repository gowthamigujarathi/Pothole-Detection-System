from ultralytics import YOLO
import cv2

# ---- CONFIG ----
MODEL_PATH = "best.pt"     # your model file
INPUT_IMAGE = "test.webp"  # put your image here
OUTPUT_IMAGE = "output.jpg"

# Load the model
model = YOLO(MODEL_PATH)

# Load image
img = cv2.imread(INPUT_IMAGE)

# Run prediction
results = model(img, conf=0.25)[0]

# Draw bounding boxes
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])

    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

    # Label
    label = f"Pothole {conf:.2f}"
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0,255,0), 2)

# Save output
cv2.imwrite(OUTPUT_IMAGE, img)
print(f"Detection complete! Output saved as: {OUTPUT_IMAGE}")
