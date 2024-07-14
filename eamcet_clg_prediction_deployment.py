import streamlit as st
import pandas as pd
import numpy as np
import pickle
import openpyxl

# Load the trained model
loaded_model = pickle.load(open('rfr_model.sav', 'rb'))

# Load the dataset
df = pd.read_excel('modified_2022.xlsx')

# Define rank columns dictionary
rank_columns = {
    'OC': ['OC_BOYS', 'OC_GIRLS'],
    'SC': ['SC_BOYS', 'SC_GIRLS'],
    'ST': ['ST_BOYS', 'ST_GIRLS'],
    'BCA': ['BCA_BOYS', 'BCA_GIRLS'],
    'BCB': ['BCB_BOYS', 'BCB_GIRLS'],
    'BCC': ['BCC_BOYS', 'BCC_GIRLS'],
    'BCD': ['BCD_BOYS', 'BCD_GIRLS'],
    'BCE': ['BCE_BOYS', 'BCE_GIRLS'],
    'OC_EWS': ['OC_EWS_BOYS', 'OC_EWS_GIRLS']
}

# Function to predict top 5 colleges
def predict_colleges(input_rank, input_gender, input_caste):
    # Filter dataset based on gender
    if input_gender == 'boys':
        df_filtered = df[df['type_of_college'] != 0]
    else:
        df_filtered = df.copy()

    # Extract relevant columns based on caste
    relevant_columns = rank_columns[input_caste]

    # Handle missing values
    df_filtered[relevant_columns] = df_filtered[relevant_columns].fillna(999999)

    # Predict rank for input
    predicted_rank = loaded_model.predict([[input_rank]])[0]

    # Find top 5 closest colleges
    distances = np.abs(df_filtered[relevant_columns].min(axis=1) - input_rank)
    top_5_indices = distances.nsmallest(5).index
    top_5_colleges = df_filtered.loc[top_5_indices]

    return top_5_colleges[['inst_name', 'branch_code']]

# Streamlit UI
st.title('College Recommendation System')

# Input form
input_rank = st.number_input('Enter your rank:', min_value=1, max_value=10000000, step=1)
input_gender = st.selectbox('Select your gender:', ['boys', 'girls'])
input_caste = st.selectbox('Select your caste:', list(rank_columns.keys()))

if st.button('Predict'):
    top_5_colleges = predict_colleges(input_rank, input_gender, input_caste)
    st.write('Top 5 Recommended Colleges:')
    st.write(top_5_colleges)
