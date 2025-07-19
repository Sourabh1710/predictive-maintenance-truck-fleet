import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import numpy as np

# Loading the pre-trained model and preprocessing objects 
try:
    model = joblib.load('src\\final_model.pkl')
    imputer = joblib.load('src\imputer.pkl')
    model_columns = joblib.load('src\model_columns.pkl')
    original_columns = joblib.load('src\original_columns.pkl')
except FileNotFoundError as e:
    st.error(f"Model file not found! Error: {e}. Please ensure the .pkl files are in the same directory as this app.py file.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred loading the model files: {e}")
    st.stop()

# Defining the Streamlit App UI 
st.set_page_config(page_title="Truck Failure Prediction", layout="wide")
st.title('🚚 Truck Fleet Predictive Maintenance')
st.write("""
This app predicts the likelihood of an Air Pressure System (APS) failure in a Scania truck. 
Upload a CSV file containing sensor readings to get a prediction for each truck (row).
""")

uploaded_file = st.file_uploader("Choose a CSV file (either clean or the original Scania format)")

if uploaded_file is not None:
    try:
        # First, try reading it as a standard CSV
        data = pd.read_csv(uploaded_file, na_values='na')
        st.info("Successfully read as a standard CSV file.")
    except (pd.errors.ParserError, pd.errors.EmptyDataError):
        # If that fails, it might be the original format with skipped rows.
        uploaded_file.seek(0)
        try:
            data = pd.read_csv(uploaded_file, skiprows=20, na_values='na')
            st.info("File has comment rows. Reading as original Scania dataset format.")
        except Exception as e:
            st.error(f"Failed to parse the file in either format. Error: {e}")
            st.stop() # Stop execution if file can't be read
    except Exception as e:
        st.error(f"An unexpected error occurred while reading the file: {e}")
        st.stop()
    


    st.write("Uploaded Data Preview:")
    st.dataframe(data.head())

    data_to_predict = data.copy()

    if 'class' in data_to_predict.columns:
        data_to_predict = data_to_predict.drop('class', axis=1)

    # Preprocessing the uploaded data
    #  Imputing missing values
    data_imputed = pd.DataFrame(imputer.transform(data_to_predict), columns=original_columns)

    #  Adding the Same feature engineering as in training
    WINDOW_SIZE = 10
    COLS_TO_ENGINEER = ['ag_005', 'ag_006', 'ay_005', 'ay_006', 'az_004']

    for col in COLS_TO_ENGINEER:
        if col in data_imputed.columns:
            data_imputed[f'{col}_roll_mean'] = data_imputed[col].rolling(window=WINDOW_SIZE, min_periods=1).mean()
            data_imputed[f'{col}_roll_std'] = data_imputed[col].rolling(window=WINDOW_SIZE, min_periods=1).std()
            data_imputed[f'{col}_roll_max'] = data_imputed[col].rolling(window=WINDOW_SIZE, min_periods=1).max()
    
    data_imputed.fillna(0, inplace=True)

    #  Aligning columns to match the model's training data
    data_processed = pd.DataFrame(columns=model_columns)
    for col in model_columns:
        if col in data_imputed.columns:
            data_processed[col] = data_imputed[col]
        else:
            data_processed[col] = 0
    
    data_processed = data_processed[model_columns]

    # Make and Display Predictions 
    st.subheader("Prediction Results")
    try:
        predictions = model.predict(data_processed)
        prediction_proba = model.predict_proba(data_processed)

        results_df = pd.DataFrame({
            'Prediction': ['🚨 FAILURE 🚨' if p == 1 else 'Normal' for p in predictions],
            'Failure Probability': prediction_proba[:, 1]
        })
        
        def style_predictions(val):
            color = 'red' if val == '🚨 FAILURE 🚨' else 'green'
            return f'color: {color}; font-weight: bold'

        st.dataframe(results_df.style.applymap(style_predictions, subset=['Prediction']).format({'Failure Probability': '{:.2%}'}))

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")