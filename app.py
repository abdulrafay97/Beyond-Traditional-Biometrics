import streamlit as st
from PIL import Image

st.set_page_config(page_title="Beyond-Traditional-Biometrics", page_icon=":lips:", layout="wide")

# Add a title to the app
st.title("Beyond-Traditional-Biometrics")

# Create two columns for displaying images side by side
col1, col2 = st.beta_columns(2)

# Ask user to upload the first image and display it in the first column
with col1:
    st.header("Image 1")
    img1_file = st.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png"])
    if img1_file is not None:
        img1 = Image.open(img1_file)
        img1 = img1.resize((400,400))
        st.image(img1, use_column_width=True)

# Ask user to upload the second image and display it in the second column
with col2:
    st.header("Image 2")
    img2_file = st.file_uploader("Upload Image 2", type=["jpg", "jpeg", "png"])
    if img2_file is not None:
        img2 = Image.open(img2_file)
        img2 = img2.resize((400,400))
        st.image(img2, use_column_width=True)
