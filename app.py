import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import time
import os
import gdown

# Download model from Google Drive
file_id = "1sI18Ii1M-wkZWkF5Asf7SnMedDIad9k4"
model_path = "Alzheimer_Model.h5"
if not os.path.exists(model_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

# Load the model
model = load_model(model_path)

# Define class labels
class_labels = ['Non Demented', 'Very Mild Dementia', 'Mild Dementia', 'Moderate Dementia']

# Preprocess image
def preprocess_image(img, target_size=(128, 128)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Predict image class
def predict_image_class(model, img):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_labels[predicted_class_idx]
    confidence = np.max(predictions)
    return predicted_class_name, confidence, predictions

# Page config
st.set_page_config(page_title="Alzheimer Stage Predictor", layout="centered", initial_sidebar_state="collapsed")

# Background color
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Logo and heading
st.markdown("<img src='https://cdn-icons-png.flaticon.com/128/8131/8131880.png' width='70' style='display: block; margin-left: auto; margin-right: auto;'>", unsafe_allow_html=True)

st.markdown("""
    <h1 style='text-align: center; color: white;'> Alzheimer Stage Predictor </h1>
    <p style='text-align: center; font-size: 18px; color: #aaa;'>Upload a brain MRI image to detect the stage of Alzheimer's disease.</p>
    <hr style='border: 1px solid #4B8BBE;'>
""", unsafe_allow_html=True)

# Upload image
uploaded_file = st.file_uploader(" Upload an MRI brain image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption=" Uploaded Image", use_container_width=True)
    st.success(" Image uploaded successfully!")

    # Fake progress
    with st.spinner("Analyzing... Please wait..."):
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1)

        predicted_class_name, confidence, predictions = predict_image_class(model, img)

    # Show result
    st.markdown(f"""
        <div style='text-align: center; padding-top: 20px;'>
            <span style='font-size: 30px; font-weight: bold; color: white;'>
                 Prediction 🧠: {predicted_class_name}
            </span>
        </div>
        <div style='text-align: center; padding-top: 10px; padding-bottom: 20px;'>
            <span style='font-size: 26px; font-weight: 600; color: white;'>
                 Confidence 📊: {confidence:.2%}
            </span>
        </div>
    """, unsafe_allow_html=True)

    # Prediction breakdown
    st.markdown("<h3 style='color:white;'> Prediction Breakdown 📈</h3>", unsafe_allow_html=True)
    for i, label in enumerate(class_labels):
        if label == predicted_class_name:
            bg_color = "#4B8BBE"
            text_color = "white"
        else:
            bg_color = "#1e1e1e"
            text_color = "white"
        st.markdown(f"""
            <div style='background-color:{bg_color}; padding:10px; margin-bottom:8px;
                        border-radius:8px; border-left:5px solid #4B8BBE; color:{text_color}'>
                <strong>{label}</strong>: {predictions[0][i]:.2%}
            </div>
        """, unsafe_allow_html=True)

    # Show visual result
    st.subheader(" Visual Result")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(
        f"Predicted: {predicted_class_name}\nConfidence: {confidence:.2%}",
        fontsize=14, fontweight='bold', color='black'
    )
    st.pyplot(fig)

    # Medical disclaimer
    st.markdown("""
      <div style='margin-top: 30px; padding: 10px; background-color: #333; border-radius: 10px; color: #ccc; text-align: center;'>
            <b> تنويه:</b> هذه النتيجة مبدئية بناءً على تحليل الصورة، ولا تُعتبر بديلاً عن الاستشارة الطبية المباشرة.
      </div>
    """, unsafe_allow_html=True)

else:
    st.info(" Please upload a brain MRI image to get started.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: grey;'>Developed by Kerolos | Powered by TensorFlow & Streamlit</p>",
    unsafe_allow_html=True
)
