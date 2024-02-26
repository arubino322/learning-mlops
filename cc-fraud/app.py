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

# maybe something like this, based on this:
# https://github.com/kylegallatin/designing-ml-systems/blob/main/unit-4/kubeflow_pipelines/kfp-notebook.ipynb
# import gcsfs
# import joblib
# import logging
# from sklearn.metrics import classification_report

# model_path = f"{base_path}/trained_models/model.joblib"
# test_features_path = f"{base_path}/preprocessed_data/test_features.csv"
# test_labels_path = f"{base_path}/preprocessed_data/test_labels.csv"

# fs = gcsfs.GCSFileSystem()

# # Load model and test data
# with fs.open(test_features_path, 'r') as f:
#     features_test_standardized = pd.read_csv(f)

# with fs.open(test_labels_path, 'r') as f:
#     target_test_encoded = pd.read_csv(f)

# # Load the model from GCS
# with fs.open(model_path, 'rb') as model_file:
#     model = joblib.load(model_file)

# predictions = model.predict(features_test_standardized)
# report = classification_report(predictions, target_test_encoded)

# # Output evaluation report
# logging.info(classification_report)