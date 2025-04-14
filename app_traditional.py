"""
Multiple Object Detection System with Streamlit Interface
Implements two classic object detection algorithms:
1. Viola-Jones Detector
2. Histogram of Oriented Gradients (HOG) Detector
"""

import cv2
import numpy as np
import streamlit as st
from skimage.feature import hog
from imutils.object_detection import non_max_suppression
from PIL import Image
import io


class MultiObjectDetector:
    def __init__(self):
        # Initialize Viola-Jones detectors
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.body_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_fullbody.xml')

        # HOG detector parameters
        self.hog_detector = cv2.HOGDescriptor()
        self.hog_detector.setSVMDetector(
            cv2.HOGDescriptor_getDefaultPeopleDetector())

    def viola_jones_detect(self, image, scale_factor=1.1, min_neighbors=5):
        """
        Detect objects using Viola-Jones cascade classifiers
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Detect bodies
        bodies = self.body_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(80, 200),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        results = {
            'faces': faces if len(faces) > 0 else np.array([]),
            'bodies': bodies if len(bodies) > 0 else np.array([])
        }

        return results

    def hog_detect(self, image, win_stride_x=8, win_stride_y=8, padding_x=16, padding_y=16, scale=1.05):
        """
        Detect objects using HOG + SVM
        """
        # Detect people using HOG
        win_stride = (win_stride_x, win_stride_y)
        padding = (padding_x, padding_y)

        (rects, weights) = self.hog_detector.detectMultiScale(
            image,
            winStride=win_stride,
            padding=padding,
            scale=scale
        )

        # If no detections, return empty array
        if len(rects) == 0:
            return np.array([])

        # Convert detections to standard format
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

        # Apply non-maxima suppression
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        # Convert back to (x, y, w, h) format
        results = []
        for (x1, y1, x2, y2) in pick:
            results.append((x1, y1, x2-x1, y2-y1))

        return np.array(results)

    def compute_hog_features(self, image_patch):
        """
        Compute HOG features for a given image patch
        """
        features = hog(
            image_patch,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False,
            transform_sqrt=True,
            block_norm='L2-Hys'
        )
        return features

    def detect_all(self, image, methods, vj_params=None, hog_params=None):
        """
        Run selected detectors and combine results
        """
        results = {}

        # Set default parameters if none provided
        if vj_params is None:
            vj_params = {"scale_factor": 1.1, "min_neighbors": 5}

        if hog_params is None:
            hog_params = {
                "win_stride_x": 8,
                "win_stride_y": 8,
                "padding_x": 16,
                "padding_y": 16,
                "scale": 1.05
            }

        if 'viola_jones' in methods:
            results['viola_jones'] = self.viola_jones_detect(
                image,
                scale_factor=vj_params["scale_factor"],
                min_neighbors=vj_params["min_neighbors"]
            )

        if 'hog' in methods:
            results['hog'] = self.hog_detect(
                image,
                win_stride_x=hog_params["win_stride_x"],
                win_stride_y=hog_params["win_stride_y"],
                padding_x=hog_params["padding_x"],
                padding_y=hog_params["padding_y"],
                scale=hog_params["scale"]
            )

        return results

    def visualize_results(self, image, results):
        """
        Draw detection results on an image
        """
        img_copy = image.copy()

        detection_count = 0

        # Draw Viola-Jones detections
        if 'viola_jones' in results:
            if results['viola_jones'].get('faces', np.array([])).size > 0:
                for face in results['viola_jones'].get('faces'):
                    x, y, w, h = face
                    cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(img_copy, 'Face (VJ)', (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    detection_count += 1

            if results['viola_jones'].get('bodies', np.array([])).size > 0:
                for body in results['viola_jones'].get('bodies'):
                    x, y, w, h = body
                    cv2.rectangle(img_copy, (x, y),
                                  (x+w, y+h), (0, 255, 255), 2)
                    cv2.putText(img_copy, 'Body (VJ)', (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    detection_count += 1

        # Draw HOG detections
        if 'hog' in results and results['hog'].size > 0:
            for rect in results['hog']:
                x, y, w, h = rect
                cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img_copy, 'Person (HOG)', (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                detection_count += 1

        return img_copy, detection_count


def run_streamlit_app():
    # Set up the Streamlit page
    st.set_page_config(page_title="Object Detector",
                       page_icon="🔍", layout="wide")

    st.title("Multiple Object Detection System")
    st.markdown("""
    This application demonstrates two classic object detection algorithms:
    1. **Viola-Jones Detector** - Fast detection of faces and bodies using Haar cascades
    2. **Histogram of Oriented Gradients (HOG)** - Effective for detecting people
    """)

    # Create sidebar for options
    st.sidebar.title("Detection Options")

    # Method selection
    st.sidebar.subheader("Select Detection Methods")
    use_viola_jones = st.sidebar.checkbox("Viola-Jones Detector", value=True)
    use_hog = st.sidebar.checkbox("HOG Detector", value=True)

    # Advanced options
    st.sidebar.subheader("Advanced Options")

    # Viola-Jones parameters
    vj_params = {}
    with st.sidebar.expander("Viola-Jones Parameters"):
        vj_params["scale_factor"] = st.slider(
            "Scale Factor", 1.05, 1.5, 1.1, 0.05)
        vj_params["min_neighbors"] = st.slider("Min Neighbors", 1, 10, 5, 1)

    # HOG parameters - using separate sliders to avoid tuple issues
    hog_params = {}
    with st.sidebar.expander("HOG Parameters"):
        hog_params["scale"] = st.slider("HOG Scale", 1.01, 1.5, 1.05, 0.01)

        # Window stride options
        stride_options = ["4", "8", "16"]
        stride_selection = st.radio(
            "Window Stride", stride_options, index=1, horizontal=True)
        hog_params["win_stride_x"] = int(stride_selection)
        hog_params["win_stride_y"] = int(stride_selection)

        # Padding options
        padding_options = ["8", "16", "24", "32"]
        padding_selection = st.radio(
            "Padding", padding_options, index=1, horizontal=True)
        hog_params["padding_x"] = int(padding_selection)
        hog_params["padding_y"] = int(padding_selection)

    # File uploader
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])

    # Initialize the detector
    detector = MultiObjectDetector()

    # Process the image when uploaded
    if uploaded_file is not None:
        # Display the original image
        pil_image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(pil_image, caption="Uploaded Image",
                     use_column_width=True)

        # Convert PIL image to OpenCV format
        image = np.array(pil_image)
        # Convert RGB to BGR (OpenCV format)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Determine which methods to use
        methods = []
        if use_viola_jones:
            methods.append('viola_jones')
        if use_hog:
            methods.append('hog')

        with st.spinner('Detecting objects...'):
            # Run detection
            results = detector.detect_all(
                image, methods, vj_params, hog_params)

            # Visualize results
            output_image, detection_count = detector.visualize_results(
                image, results)
            # Convert BGR back to RGB for display
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

        # Display the processed image
        with col2:
            st.subheader("Detection Results")
            st.image(
                output_image, caption=f"Detected {detection_count} objects", use_column_width=True)

        # Display detection details
        st.subheader("Detection Details")

        if 'viola_jones' in results:
            vj_faces = len(results['viola_jones'].get('faces', np.array([])))
            vj_bodies = len(results['viola_jones'].get('bodies', np.array([])))
            st.write(
                f"Viola-Jones detected {vj_faces} faces and {vj_bodies} bodies")

        if 'hog' in results:
            hog_people = len(results['hog'])
            st.write(f"HOG detected {hog_people} people")

        # Add download button for processed image
        if detection_count > 0:
            # Convert back to PIL for download
            result_pil = Image.fromarray(output_image)

            # Create a download button
            buf = io.BytesIO()
            result_pil.save(buf, format="PNG")
            btn = st.download_button(
                label="Download Result Image",
                data=buf.getvalue(),
                file_name="detection_result.png",
                mime="image/png"
            )
    else:
        # Show sample instructions when no image is uploaded
        st.info("Please upload an image to begin object detection")

        # Optional: Add a demo image
        if st.button("Use Demo Image"):
            # Use a built-in example image if available
            try:
                from skimage import data
                demo_image = data.astronaut()
                st.session_state.demo_image = demo_image
                st.experimental_rerun()
            except Exception:
                st.error(
                    "Could not load demo image. Please upload your own image.")

        # Check if demo image is in session state
        if 'demo_image' in st.session_state:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(st.session_state.demo_image)
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Demo Image")
                st.image(pil_image, caption="Demo Image",
                         use_column_width=True)

            # Convert PIL image to OpenCV format
            image = np.array(pil_image)
            # Already RGB, convert to BGR (OpenCV format)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Determine which methods to use
            methods = []
            if use_viola_jones:
                methods.append('viola_jones')
            if use_hog:
                methods.append('hog')

            with st.spinner('Detecting objects...'):
                # Run detection
                results = detector.detect_all(
                    image, methods, vj_params, hog_params)

                # Visualize results
                output_image, detection_count = detector.visualize_results(
                    image, results)
                # Convert BGR back to RGB for display
                output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

            # Display the processed image
            with col2:
                st.subheader("Detection Results")
                st.image(
                    output_image, caption=f"Detected {detection_count} objects", use_column_width=True)

            # Display detection details
            st.subheader("Detection Details")

            if 'viola_jones' in results:
                vj_faces = len(results['viola_jones'].get(
                    'faces', np.array([])))
                vj_bodies = len(results['viola_jones'].get(
                    'bodies', np.array([])))
                st.write(
                    f"Viola-Jones detected {vj_faces} faces and {vj_bodies} bodies")

            if 'hog' in results:
                hog_people = len(results['hog'])
                st.write(f"HOG detected {hog_people} people")


if __name__ == "__main__":
    run_streamlit_app()
