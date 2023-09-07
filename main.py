import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background

##NOTE:
set_background('./Triple_1300Lede.jpg')

# set title
st.title(':violet[Gravity Spy Image Classification]')

# set header
st.header(':orange[Please upload an image to classify]\nClick the following link to access the dataset which was used to train this model for more examples: https://www.kaggle.com/datasets/tentotheminus9/gravity-spy-gravitational-waves')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
@st.cache_resource
def load_model(model_path):
    modelresnet = tf.keras.models.load_model(model_path)
    return modelresnet

modelresnet = load_model('./gspyrsntl-20230816T183607Z-001/gspyrsntl')

# load class names
class_names =  ['Wandering Line', 'Power Line', 'Tomte', 'Helix', 'Light Modulation', 'No Glitch', 'Whistle', '1400 Ripples', 'Scratchy', 'Scattered Light', 'Air Compressor', 'Repeating Blips', 'Paired Doves', 'Violin Mode', 'Blip', 'Low Frequency Lines', 'Chirp', 'Extremely Loud', 'Low Frequency Burst', 'Could not be identified', 'Koi Fish', '1080 Lines']

# display image
if file is not None:
    image = Image.open(file).convert("RGB")
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, modelresnet, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### Confidence score: {}%".format(int(conf_score * 10000) / 100))

