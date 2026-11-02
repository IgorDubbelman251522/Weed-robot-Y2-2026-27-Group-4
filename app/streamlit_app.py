import streamlit as st
from PIL import Image

st.set_page_config(page_title="Weed robot", layout="wide")
st.title("Weed robot: image classifier")

upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if upload:
    st.image(Image.open(upload), caption=upload.name, use_container_width=True)
    st.info("Model inference is not wired up yet.")