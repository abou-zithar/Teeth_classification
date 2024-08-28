import streamlit as st
import tensorflow as tf
from PIL import Image
from predictor import predict_with_model

import numpy as np
from tensorflow.keras.preprocessing import image as keras_image



def load_model():
  model=tf.keras.models.load_model('model.keras')
  if model == None:
     print("the file is not found")
  return model




with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Teeth Classification
         """
         )

file = st.file_uploader("Please upload a Teeth image", type=["jpg", "png"])



st.set_option('deprecation.showfileUploaderEncoding', False)




if file is None:
    st.text("Please upload an image file")
else:
 
  # Open the image file using PIL
  image = Image.open(file)
  st.image(image, use_column_width=True)

  # Make prediction
  predictions, confidence = predict_with_model(model, image)

  # Display the prediction and confidence
  st.write(f"Prediction = {predictions}")
  st.write(f"This image most likely belongs to class {predictions} with a {confidence * 100:.2f}% confidence.")

