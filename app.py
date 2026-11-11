
import streamlit as st
import cv2
import numpy as np
import io
from ultralytics import YOLO
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Pothole Detection System",
    page_icon="🚧",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Application title
st.title("🚧 Pothole Detection System")
st.write("AI-powered road pothole detection using YOLOv8")

st.divider()

# Upload image
uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    st.subheader("Original Image")
    st.image(image_array, use_container_width=True)

    if st.button("Detect Potholes", type="primary"):

        with st.spinner("Detecting potholes..."):

            results = model(image_array, conf=0.25)[0]

            detected_image = results.plot()
            detected_image = cv2.cvtColor(
                detected_image,
                cv2.COLOR_BGR2RGB
            )

            pothole_count = len(results.boxes)

        st.success("Detection completed!")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Detection Result")
            st.image(
                detected_image,
                use_container_width=True
            )

        with col2:
            st.subheader("Detection Summary")
            st.metric("Total Potholes Detected", pothole_count)

            if pothole_count > 0:
                st.write("**Confidence Scores**")

                for i, box in enumerate(results.boxes):
                    confidence = float(box.conf[0])
                    st.write(
                        f"Pothole {i+1}: {confidence:.2%}"
                    )
            else:
                st.info("No potholes detected.")

        result_image = Image.fromarray(detected_image)

        buffer = io.BytesIO()
        result_image.save(buffer, format="PNG")

        st.download_button(
        label="Download Detection Result",
        data=buffer.getvalue(),
        file_name="pothole_detection.png",
        mime="image/png"
        )

st.divider()
st.caption("Pothole Detection System | YOLOv8 | Deep Learning")