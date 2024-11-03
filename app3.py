import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image, ImageOps
import tempfile
import cv2

# Page configuration
st.set_page_config(page_title="3D Printing Defect Detector", page_icon="🦾", layout="wide")

# Custom styles for light blue background and black font
st.markdown(
    """
    <style>
    body {
        background-color: #d9f2ff;
        color: #000000;
        font-family: 'Segoe UI', sans-serif;
    }
    .main-header {
        text-align: center;
        padding: 20px;
        background-color: #b3e0ff;
        color: #000000;
        border-radius: 10px;
    }
    .main-header h1 {
        font-size: 3em;
        font-weight: 700;
    }
    .sidebar .element-container {
        color: #000000;
    }
    .result-box {
        text-align: center;
        font-size: 1.5em;
        font-weight: 600;
        padding: 15px;
        color: #000000;
        border-radius: 8px;
    }
    .result-box.defective {
        background-color: #ff4d4f;
    }
    .result-box.no-defect {
        background-color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown('<div class="main-header"><h1>3D Printing Defect Detector </h1></div>', unsafe_allow_html=True)

# Load model function
def load_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    with open('cnn_model_weights.sav', 'rb') as f:
        weights = pickle.load(f)
        model.set_weights(weights)
    return model

# Image preprocessing
def preprocess_image(image):
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    image = ImageOps.invert(image)
    image = np.array(image) / 255.0
    return image.reshape(1, 28, 28, 1)

# Video preprocessing
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (28, 28))
        inverted_frame = cv2.bitwise_not(resized_frame)
        frames.append(inverted_frame / 255.0)
        if len(frames) == 10:
            break
    cap.release()
    return np.array(frames).reshape(-1, 28, 28, 1)

# Sidebar with input selection
with st.sidebar:
    st.title("Customize Detection")
    option = st.radio("Select Input", ["Image", "Video"], index=0)

model = load_model()

# Main display area
if option == "Image":
    uploaded_file = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])

# Display the uploaded image at a reduced size
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        resized_image = image.resize((300, 300))  # Resize to 300x300 pixels or any size you prefer
        st.image(resized_image, caption="Uploaded Image (Resized)", use_column_width=False)
     
        # Process image and make prediction
        with st.spinner("Processing Image..."):
            preprocessed_image = preprocess_image(image)
            prediction = model.predict(preprocessed_image)
            predicted_defect = np.argmax(prediction, axis=1)[0]
        
        # Display result
        result_text = "Defective" if predicted_defect else "No Defect Detected"
        result_class = "defective" if predicted_defect else "no-defect"
        st.markdown(f'<div class="result-box {result_class}">{result_text}</div>', unsafe_allow_html=True)

elif option == "Video":
    uploaded_file = st.file_uploader("Upload a video:", type=["mp4"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            st.video(uploaded_file, format="video/mp4", start_time=0)
            
            # Process video frames and make prediction
            with st.spinner("Processing Video..."):
                frames = preprocess_video(tmp_file.name)
                predictions = model.predict(frames)
                defect_counts = np.sum(np.argmax(predictions, axis=1))
        
        # Display result
        result_text = "Defective" if defect_counts > len(frames) * 0.5 else "No Defect Detected"
        result_class = "defective" if defect_counts > len(frames) * 0.5 else "no-defect"
        st.markdown(f'<div class="result-box {result_class}">{result_text}</div>', unsafe_allow_html=True)

# Project Overview section
st.markdown(
    """
    <hr>
    <h2 style="text-align: center; color: #000000;">Project Overview</h2>
    <p style="color: #000000;">
        This app detects potential defects in 3D printed objects using a CNN model. By analyzing images or videos, the app helps to
        identify defective prints in real-time, minimizing material waste and improving printing efficiency.
    </p>
    """,
    unsafe_allow_html=True
)
