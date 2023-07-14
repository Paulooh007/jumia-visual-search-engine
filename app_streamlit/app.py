import sys
from pathlib import Path


import streamlit as st
from image_search_engine import utils
from image_search_engine.models import EfficientNet_b0_ns

col1, col2, col3 = st.columns(3)


with col1:
    st.header("A cat")
    st.image("https://static.streamlit.io/examples/cat.jpg")

    st.header("A cat")
    st.image("https://static.streamlit.io/examples/cat.jpg")

with col2:
    st.header("A dog")
    st.image("https://static.streamlit.io/examples/dog.jpg")

with col3:
    st.header("An owl")
    st.image("https://static.streamlit.io/examples/owl.jpg")
