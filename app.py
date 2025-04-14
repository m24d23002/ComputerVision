# app.py
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from pycocotools.coco import COCO
from train import buildModel
import os

# Load COCO category mappings
annotation_file = 'instances_val2017.json'
coco = COCO(annotation_file)
cat_ids = sorted(coco.getCatIds())
index_to_cat_id = {i: cat_id for i, cat_id in enumerate(cat_ids)}
cat_id_to_name = {cat['id']: cat['name'] for cat in coco.loadCats(cat_ids)}



# Cache the model creation
@st.cache_resource
def get_model():
    return buildModel()

# Create the model
model = get_model()

def detect_objects(model, image):
    image_resized = cv2.resize(image, (224, 224))  # Resize to model's input size
    image_resized = image_resized / 255.0  # Normalize
    image_input = np.expand_dims(image_resized, axis=0)

    # Perform object detection
    class_predictions, bbox_predictions = model.predict(image_input)

    # Assuming bbox_predictions are normalized bounding boxes relative to input size
    for i in range(len(bbox_predictions[0])):
        x, y, w, h = bbox_predictions[0][i]
        class_id = np.argmax(class_predictions[0])  # Get predicted class ID
        confidence = np.max(class_predictions[0])  # Get confidence score

        # Draw bounding box
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

        # Get the class name from the COCO dataset
        class_name = cat_id_to_name.get(class_id, f"Class {class_id}")
        label = f"{class_name}: {confidence:.2f}"

        # Display the label
        cv2.putText(image, label, (int(x), int(y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return image

# Streamlit App
def main():
    st.title("Custom Object Detection with COCO Dataset")

    choice = st.selectbox("Choose an option", ["Upload your image"])

    if choice == "Upload your image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)

            # Detect objects
            detected_image = detect_objects(model, image_np)

            # Display the original image with bounding boxes
            st.image(detected_image, channels="RGB", caption="Detected Objects", use_column_width=True)

if __name__ == "__main__":
    main()
