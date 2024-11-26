import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model("skin_cancer_model.h5")

# Define image size
IMG_SIZE = (128, 128)

# Preprocessing function
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = np.asarray(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Prediction function
def predict_image(image, model):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    return predictions


# Prediction function
def predict_image(image, model):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    return predictions



st.title("Skin Cancer Detection App")
st.write("Upload an image to check for skin cancer.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

class_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']  # Original class names
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Make predictions
    st.write("Processing the image...")
    predictions = predict_image(image, model)

    # Show prediction results
     # Map predictions to class labels
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_labels[predicted_class_index]

    st.write(f"Predicted Class: {predicted_class_name}")
    st.write("Prediction Probabilities:")
    for label, prob in zip(class_labels, predictions[0]):
        st.write(f"{label}: {prob:.4f}")

    
    # class_labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6']
    # predicted_class = np.argmax(predictions)
    # st.write(f"Predicted Class: {class_labels[predicted_class]}")
    # st.write("Prediction Probabilities:")
    # for label, prob in zip(class_labels, predictions[0]):
    #     st.write(f"{label}: {prob:.4f}")

