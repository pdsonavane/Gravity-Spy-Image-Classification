import base64
import tensorflow as tf

import streamlit as st
from PIL import ImageOps, Image
import numpy as np


def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


def classify(image, model, class_names):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
        samplewise_center=True,
        samplewise_std_normalization=True
    )
    # Load the image and apply preprocessing
    #image = tf.keras.preprocessing.image.load_img(image, target_size=target_size)
    #image=image.resize(224,224) #our implementation
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    preprocessed_image = datagen.standardize(image_array[np.newaxis, ...])
    predictions = model.predict(preprocessed_image)
    #print(predictions)
    
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    #print(predicted_class_index)
    class_name = class_names[predicted_class_index]
    confidence_score = predictions[0][predicted_class_index] 
    return class_name, confidence_score