import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D

from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
from PIL import Image
import streamlit as st

# image features and file names:
Image_features = pkl.load(open("Image_features.pkl", "rb"))
filenames = pkl.load(open("filenames.pkl", "rb"))

# models
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.models.Sequential([model, GlobalMaxPool2D()])

# Creating the app:
st.header("Fashion Recommendation System")


##Step 2:Save the file


def save_uploaded_file(uploaded_file):
    try:
        file_path = os.path.join("upload", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


##Step 4: Create a function for feature extraction


def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result / norm(result)

    return norm_result


##Step 5: Create a function for recommendation


def recommend(features, Image_features):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm="brute", metric="euclidean")
    neighbors.fit(Image_features)
    distance, indices = neighbors.kneighbors([features])

    # the indices will be calculated and send back to the calling function
    return indices


##Step 1:Upload the file

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Save the uploaded image to a  directory
    if save_uploaded_file(uploaded_file):
        ##Step 3: display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        # extract features
        features = feature_extraction(os.path.join("upload", uploaded_file.name), model)
        st.text(features)
        indices = recommend(features, Image_features)
        ##Step 6: Display the indices
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][1]])
        with col2:
            st.image(filenames[indices[0][2]])
        with col3:
            st.image(filenames[indices[0][3]])
        with col4:
            st.image(filenames[indices[0][4]])
        with col5:
            st.image(filenames[indices[0][5]])

    else:
        st.header("Some error occured in file upload")
