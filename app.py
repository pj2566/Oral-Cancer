
import pickle
import streamlit as st
import tensorflow as tf
import streamlit as st
class_names=['Oscc', 'Normal']


@st.cache(allow_output_mutation=True)
def load_model():
  model= pickle.load(open("Oral Cancer detct.pkl", "rb"))
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Cancer Classification
         """
         )

file = st.file_uploader("Please upload an Cancer Photo", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    size = (256,256)
    image = Image.open(file)
    resize=ImageOps.fit(image, size, Image.LANCZOS)
    resize = np.asarray(resize) 
   
    img_reshape = resize[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction,image
    
    
       
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions,image = import_and_predict(image, model)
    
    st.write("Normal- {:.8f}".format(float(predictions[0][0]*100)))
    st.write("OSCC - {:.8f}".format(float(predictions[0][1]*100)))
    st.image(image, caption='Pic')
 
    

