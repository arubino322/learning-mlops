import streamlit as st
import pandas as pd
from io import StringIO

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)

# TODO:
# what do i want this file to do?
# read input from a file (csv)
# read model saved in model directory (or for first past, saved to GCS)
# print output of the predictions
