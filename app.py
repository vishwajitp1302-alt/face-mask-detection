import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model

import importlib

try:
    st_cropper = importlib.import_module("streamlit_cropper").st_cropper
except Exception:
    st_cropper = None

st.title("Face Mask Detection Project")
model = load_model("face-mask-detector.keras")

option = st.selectbox("Select", ["Image", "Capture"])
if option == "Image":
    camera_image = None
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
else:
    camera_image = st.camera_input("Capture a photo")
    uploaded_file = None

    if camera_image:
        image = Image.open(camera_image)
        if st_cropper:
            st.write("Crop the image")
            cropped_img = st_cropper(
                image,
                realtime_update=True,
                box_color='red',
                aspect_ratio=(1, 1)
            )
        else:
            st.warning("streamlit_cropper not installed, using full captured image")
            cropped_img = image

        # Resize for CNN
        cropped_img = cropped_img.resize((150, 150))
        st.image(cropped_img, caption="Cropped Image")


if st.button("Detect"):
    image_to_detect = None
    if uploaded_file is not None:
        st.image(uploaded_file)
        image_to_detect = Image.open(uploaded_file)
    elif camera_image is not None:
        image_to_detect = cropped_img
    else:
        st.error("Please upload an image")

    if image_to_detect is not None:
        image_to_detect = image_to_detect.convert("RGB")
        img = image_to_detect.resize((150, 150))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        result = model.predict(img_array)
        if result[0, 0] <= 0.5:
            st.success("Person is with mask")
        else:
            st.error("Person is without mask")