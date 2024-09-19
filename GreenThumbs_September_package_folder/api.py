
#-------------------------------------------------------------------------------
# Author : Thierry CAROLE
# Batch Le Wagon 1575 - Data Science
# Date : 19/09/2024
# Github link : https://github.com/thcarole1/GreenThumbs_September
# Code from GreenThumbs_September_package_folder/api.py
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# GLOBAL STRATEGY : After processing the input (raw reviews from json file),
# the API will return the corresponding predictions (A note over 10 i.e 2/10 or 9/10, ...).
#-------------------------------------------------------------------------------

# Basic libraries
import pandas as pd
import numpy as np
import json

#Import FastApi python framework
from fastapi import FastAPI, UploadFile, File

# Import custom functions from .py files
from GreenThumbs_September_package_folder.api_functions.preprocessor_api import preprocess_features, get_tokenized,get_padded
from GreenThumbs_September_package_folder.api_functions.model_api import load_tokenizer,load_model,get_prediction

# Instantiate FastAPI
app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'Certification Data Science Le Wagon': 'Session de Septembre 2024 !'}

# GLOBAL STRATEGY : After processing the input (raw reviews from json file),
# the API will return the corresponding predictions.
# Define an endpoint allowing to upload a json file (i.e raw reviews).
@app.post("/upload_and_predict_sentiment")
def create_upload_files(upload_file: UploadFile = File(...)):
    # Retrieve input data from json file
    json_data = json.load(upload_file.file)

    # Create a Pandas dataframe from json input file
    # Type of X_test : <class 'pandas.core.frame.DataFrame'>
    X_test = pd.DataFrame(json_data)
    print(f"✅ Type of X_test : {type(X_test)}")

    # Preprocess data (i.e Removing whitespaces, Lowercasing,
    # Removing numbers, Removing punctuation, Lemmatizing)
    #Type of X_test_preproc : <class 'pandas.core.series.Series'>
    X_test_preproc = preprocess_features(X_test['ReviewText'])
    print(f"✅ Type of X_test_preproc : {type(X_test_preproc)}")

    # ---- Tokenize the preprocessed data----
    #First, we retrieve the tokenizer that has been fitting on the
    # training dataset previously (fitted tokenizer stored on the cloud for example).
    #Type of tokenizer : <class 'keras.src.legacy.preprocessing.text.Tokenizer'>
    tokenizer = load_tokenizer()
    print(f"✅ Type of tokenizer : {type(tokenizer)}")

    # Secondly, we tokenize the preprocessed data with the fitted tokenizer
    #Type of X_test_tokens : <class 'list'>
    X_test_tokens = get_tokenized(X_test_preproc, tokenizer)
    print(f"✅ Type of X_test_tokens : {type(X_test_tokens)}")

    # Padding the preprocessed and tokenized data
    #Type of X_test_pad : <class 'numpy.ndarray'>
    X_test_pad = get_padded(X_test_tokens, maxlen = 30)
    print(f"✅ Type of X_test_pad : {type(X_test_pad)}")

    #  Load RNN model that is already fitted on the training data.
    #(fitted RNN model stored on the cloud for example).
    #Type of model : <class 'keras.src.models.sequential.Sequential'>
    model = load_model('RNN')
    print(f"✅ Type of model : {type(model)}")

    # Prediction
	# For information, the get_prediction function is described below
    #Type of prediction : <class 'numpy.ndarray'>
    prediction = get_prediction(X_test_pad, model)
    print(f"✅ Type of prediction : {type(prediction)}")
    prediction = prediction.tolist()

    #Type of prediction_final : <class 'list'>
    prediction_final = [np.round(pred[0],1)*10 for pred in prediction]
    print(f"✅ Type of prediction_final : {type(prediction_final)}")
    print(prediction_final)
    # Return the prediction to the user
    return {'Prediction' : prediction_final}
