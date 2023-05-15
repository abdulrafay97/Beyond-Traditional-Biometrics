import streamlit as st
from PIL import Image
import torch
import helper
import numpy as np
from torchvision import transforms
import cv2
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Beyond-Traditional-Biometrics", page_icon=":lips:", layout="wide")

#Load Yolo
model_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path='./Weights/best.pt')

cropped_img1 = np.zeros((1, 1, 3), dtype=np.uint8)
cropped_img2 = np.zeros((1, 1, 3), dtype=np.uint8)

def on_button_click(img1_vector, img2_vector):
  cos_sim = cosine_similarity(img1_vector.reshape(1, -1), img2_vector.reshape(1, -1))
  return cos_sim[0][0]

col1, col2 = st.columns(2)
with col1:
  st.header("Image 1")
  img1_file_up = st.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png"])
  if img1_file_up is not None:
    file_bytes1 = np.asarray(bytearray(img1_file_up.read()), dtype=np.uint8)
    img1 = cv2.imdecode(file_bytes1, cv2.IMREAD_COLOR)
    st.image(img1,caption = 'Uploaded Image 1.', width=300, use_column_width=300)
    cropped_img1 = helper.getting_Lips(img1, model_yolo)

with col2:
  st.header("Image 2")
  img2_file_up = st.file_uploader("Upload Image 2", type=["jpg", "jpeg", "png"])
  if img2_file_up is not None:
    file_bytes2 = np.asarray(bytearray(img2_file_up.read()), dtype=np.uint8)
    img2 = cv2.imdecode(file_bytes2, cv2.IMREAD_COLOR)
    st.image(img2,caption = 'Uploaded Image 2.', width=300, use_column_width=300)
    cropped_img2 = helper.getting_Lips(img2, model_yolo)

l_col1, l_col2 = st.columns(2)
with l_col1:
  st.header("Extracted Lips For Image 1")
  st.image(cropped_img1, caption = 'Extracted Lips.', width=300, use_column_width=300)
  img1_vector = helper.preprocessing(cropped_img1)

with l_col2:
  st.header("Extracted Lips For Image 2")
  st.image(cropped_img2, caption = 'Extracted Lips.', width=300, use_column_width=300)
  img2_vector = helper.preprocessing(cropped_img2)

if st.button("Find Similarity"):
  cosine_Score = on_button_click(img1_vector, img2_vector)
  xy = round(cosine_Score, 1)
  if xy > 0.7:
    st.write(f"Its a Match with {xy} Cosine Similarity!")
  else:
    st.write(f"Its not Match with {xy} Cosine Similarity!")
  