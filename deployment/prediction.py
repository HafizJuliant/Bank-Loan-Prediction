import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

with open('model_svm.pkl', 'rb') as file_1:
  model_svm_grid = pickle.load(file_1)

def run():

    with st.form('form_loan_status'):
        Loan_ID = st.text_input('Loan ID', value=' ')
        Gender= st.selectbox('Gender', ('Male','Female'))
        Married= st.selectbox('Married',('0','1'), help=('1= Yes, 0= No'))
        Dependents= st.selectbox('Dependents', ('0','1','2','3'))
        Education= st.selectbox('Education', ('Graduated','Undergraduated'))
        Self_Employed= st.selectbox('Self Employed', ('Yes','No'))
        ApplicantIncome= st.number_input('Applicant Income', min_value=150, max_value=100000, value=1000, help=('value in USD'))
        CoapplicantIncome= st.number_input('Co applicant Income', min_value=0, max_value=100000, value=0, help=('value in USD'))
        LoanAmount= st.number_input('Loan Amount', min_value=5 , max_value=10000, value=100, help=('value in USD'))
        Loan_Amount_Term= st.number_input('Loan Amount Term', min_value= 12, max_value=480, value=360, help=('value in months'))
        Credit_History= st.selectbox('Credit History', ('0','1'), help=('1= Yes, 0= No'))
        Property_Area= st.selectbox('Property_Area',('Rural','Urban','Semiurban'))

        submitted = st.form_submit_button('Predict')

    data_inf = {
        'Loan_ID': Loan_ID,
        'Gender': Gender,
        'Married': Married,
        'Dependents': Dependents,
        'Education': Education,
        'Self_Employed': Self_Employed,
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount,
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': Credit_History,
        'Property_Area': Property_Area,
        }
    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)  

    if submitted:  
       y_pred_inf = model_svm_grid.predict(data_inf)

       st.write('## Loan Status:')

       for pred in y_pred_inf:
            if pred == 1:
                st.write('Loan Approved')
            else:
                st.write('Loan Not Approved')

if __name__ == '__main__':
    run()

       