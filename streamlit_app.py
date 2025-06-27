import streamlit as st
import requests
import pandas as pd

st.title("Thyroid Disease Prediction")

# Define categorical options based on dataset
categorical_options = {
    'sex': ['M', 'F'],
    'on_thyroxine': ['t', 'f'],
    'query_on_thyroxine': ['t', 'f'],
    'on_antithyroid_meds': ['t', 'f'],
    'sick': ['t', 'f'],
    'pregnant': ['t', 'f'],
    'thyroid_surgery': ['t', 'f'],
    'I131_treatment': ['t', 'f'],
    'query_hypothyroid': ['t', 'f'],
    'query_hyperthyroid': ['t', 'f'],
    'lithium': ['t', 'f'],
    'goitre': ['t', 'f'],
    'tumor': ['t', 'f'],
    'hypopituitary': ['t', 'f'],
    'psych': ['t', 'f'],
    'TSH_measured': ['t', 'f'],
    'T3_measured': ['t', 'f'],
    'TT4_measured': ['t', 'f'],
    'T4U_measured': ['t', 'f'],
    'FTI_measured': ['t', 'f'],
    'TBG_measured': ['t', 'f'],
    'referral_source': ['SVHC', 'SVI', 'STMW', 'SVHD', 'other']
}

# Input form
with st.form(key='thyroid_form'):
    st.subheader("Patient Information")
    age = st.number_input("Age", min_value=1, max_value=100, value=50)
    sex = st.selectbox("Sex", options=categorical_options['sex'])
    
    st.subheader("Medical History")
    on_thyroxine = st.selectbox("On Thyroxine", options=categorical_options['on_thyroxine'])
    query_on_thyroxine = st.selectbox("Query On Thyroxine", options=categorical_options['query_on_thyroxine'])
    on_antithyroid_meds = st.selectbox("On Antithyroid Meds", options=categorical_options['on_antithyroid_meds'])
    sick = st.selectbox("Sick", options=categorical_options['sick'])
    pregnant = st.selectbox("Pregnant", options=categorical_options['pregnant'])
    thyroid_surgery = st.selectbox("Thyroid Surgery", options=categorical_options['thyroid_surgery'])
    I131_treatment = st.selectbox("I131 Treatment", options=categorical_options['I131_treatment'])
    query_hypothyroid = st.selectbox("Query Hypothyroid", options=categorical_options['query_hypothyroid'])
    query_hyperthyroid = st.selectbox("Query Hyperthyroid", options=categorical_options['query_hyperthyroid'])
    lithium = st.selectbox("Lithium", options=categorical_options['lithium'])
    goitre = st.selectbox("Goitre", options=categorical_options['goitre'])
    tumor = st.selectbox("Tumor", options=categorical_options['tumor'])
    hypopituitary = st.selectbox("Hypopituitary", options=categorical_options['hypopituitary'])
    psych = st.selectbox("Psych", options=categorical_options['psych'])
    
    st.subheader("Lab Measurements")
    TSH_measured = st.selectbox("TSH Measured", options=categorical_options['TSH_measured'])
    TSH = st.number_input("TSH", min_value=0.0, value=0.0, step=0.1) if TSH_measured == 't' else None
    T3_measured = st.selectbox("T3 Measured", options=categorical_options['T3_measured'])
    T3 = st.number_input("T3", min_value=0.0, value=0.0, step=0.1) if T3_measured == 't' else None
    TT4_measured = st.selectbox("TT4 Measured", options=categorical_options['TT4_measured'])
    TT4 = st.number_input("TT4", min_value=0.0, value=0.0, step=0.1) if TT4_measured == 't' else None
    T4U_measured = st.selectbox("T4U Measured", options=categorical_options['T4U_measured'])
    T4U = st.number_input("T4U", min_value=0.0, value=0.0, step=0.1) if T4U_measured == 't' else None
    FTI_measured = st.selectbox("FTI Measured", options=categorical_options['FTI_measured'])
    FTI = st.number_input("FTI", min_value=0.0, value=0.0, step=0.1) if FTI_measured == 't' else None
    TBG_measured = st.selectbox("TBG Measured", options=categorical_options['TBG_measured'])
    referral_source = st.selectbox("Referral Source", options=categorical_options['referral_source'])
    
    submit_button = st.form_submit_button(label='Predict')

# Make prediction when form is submitted
if submit_button:
    # Prepare input data
    input_data = {
        'age': age,
        'sex': sex,
        'on_thyroxine': on_thyroxine,
        'query_on_thyroxine': query_on_thyroxine,
        'on_antithyroid_meds': on_antithyroid_meds,
        'sick': sick,
        'pregnant': pregnant,
        'thyroid_surgery': thyroid_surgery,
        'I131_treatment': I131_treatment,
        'query_hypothyroid': query_hypothyroid,
        'query_hyperthyroid': query_hyperthyroid,
        'lithium': lithium,
        'goitre': goitre,
        'tumor': tumor,
        'hypopituitary': hypopituitary,
        'psych': psych,
        'TSH_measured': TSH_measured,
        'TSH': TSH,
        'T3_measured': T3_measured,
        'T3': T3,
        'TT4_measured': TT4_measured,
        'TT4': TT4,
        'T4U_measured': T4U_measured,
        'T4U': T4U,
        'FTI_measured': FTI_measured,
        'FTI': FTI,
        'TBG_measured': TBG_measured,
        'referral_source': referral_source
    }
    
    # Call FastAPI endpoint
    try:
        response = requests.post("http://localhost:8000/predict", json=input_data)
        if response.status_code == 200:
            prediction = response.json()['prediction']
            st.success(f"Prediction: {prediction}")
        else:
            st.error(f"Error: {response.json()['detail']}")
    except Exception as e:
        st.error(f"Failed to connect to API: {str(e)}")