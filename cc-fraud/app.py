import gcsfs
import pickle

import streamlit as st
import pandas as pd

from sklearn.metrics import classification_report

base_path = "gs://machine-learning-workspace"
model_path = f"{base_path}/cc-fraud/models/clf_model.pkl"
test_features_path = f"{base_path}/cc-fraud/data/sample_features_test_v2.csv"

fs = gcsfs.GCSFileSystem()

# Load the model from GCS
with fs.open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# with fs.open(test_features_path, 'r') as f:
#     features_test = pd.read_csv(f)

# predictions = model.predict(features_test)
# print(predictions)
# instead of printing predictions, do we want to evaluate data?
# report = classification_report(predictions, target_test_encoded)

with fs.open(f"{base_path}/cc-fraud/data/eval/sample_test.csv", 'r') as f:
    sample_test = pd.read_csv(f)

feats = sample_test.iloc[:, 1:30]
targets = sample_test['Class']
predictions = model.predict(feats)
print(classification_report(predictions, targets))

# when inputing the file, should I include actual target var so I can monitor accuracy?

# data/eval/ is where input data from streamlit will go
# data/predictions is where predictions will go (with eval data?) 



# uploaded_file = st.file_uploader("Choose a file")
# if uploaded_file is not None:
#     # Can be used wherever a "file-like" object is accepted:
#     # Load test data (without labels)
#     features_test_standardized = pd.read_csv(uploaded_file)
#     st.write(features_test_standardized)

#     # Load model and test data
#     # with fs.open(test_features_path, 'r') as f:
#     #     features_test_standardized = pd.read_csv(f)

#     # with fs.open(test_labels_path, 'r') as f:
#     #     target_test_encoded = pd.read_csv(f)

#     predictions = model.predict(features_test_standardized)
#     # print(predictions)
#     st.write(predictions)
#     # report = classification_report(predictions, target_test_encoded)
#     # your are comparing your predictions with the actual target laels

#     # Output evaluation report
#     # logging.info(classification_report)

# TODO:
# figure out this error: FileNotFoundError: b/machine-learning-workspace/o 
# (gcloud config set project PROJECT_ID)
# read input from a file (csv)
# read model saved in model directory (or for first past, saved to GCS)
# print output of the predictions

# maybe some thing like this, based on this:
# https://github.com/kylegallatin/designing-ml-systems/blob/main/unit-4/kubeflow_pipelines/kfp-notebook.ipynb

# I think I chose to use pandas b/c it plays nice with Streamlit
# However, the end result I might not need it, since streamlit will simply be a UI to add more input
# data to then trigger/retrain the model.

