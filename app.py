import streamlit as st
from PIL import Image

st.set_page_config(page_title="Image Comparison", page_icon=":lips:", layout="wide")

# Add a title to the app
st.title("Image Comparison")

# Create two columns for displaying images side by side
col1, col2 = st.beta_columns(2)

# Load the first image and display it in the first column
with col1:
    st.header("Image 1")
    img1 = Image.open("./10_1.jpg")
    st.image(img1, use_column_width=True)

# Load the second image and display it in the second column
with col2:
    st.header("Image 2")
    img2 = Image.open("./21_1.jpg")
    st.image(img2, use_column_width=True)
