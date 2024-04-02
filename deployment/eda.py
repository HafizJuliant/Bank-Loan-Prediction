import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image


def run() :
    st.title('Loan Status Prediction')
    st.subheader('EDA untuk analisa dataset Loan Status')
    #deskripsi
    st.write('**Created by Hafiz J**')

    st.markdown('-------')

    #Show dataframe
    df = pd.read_csv('deploy.csv')
    df_ori = pd.read_csv('loan_data.csv')

    st.write("## Heatmap Correlation")
    correlation_matrix = df.drop('Loan_ID', axis=1).corr()
    fig = plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 10})
    heatmap.set_title("Correlation Matrix Heatmap", fontsize=16)
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Correlation', rotation=270, labelpad=15)
    st.pyplot(fig)

    st.write("## Distribution of Loan Status")
    fig1 = plt.figure(figsize=(6, 6))
    sns.barplot(x=df_ori['Loan_Status'].value_counts().index, y=df_ori['Loan_Status'].value_counts().values, palette='husl')
    st.pyplot(fig1)
    
    st.write("## Distribution of Gender, Married and Loan Status")
    fig2 = plt.figure(figsize=(10, 8))
    plot = plt.subplot(1, 2, 1)
    sns.countplot(data=df_ori, x='Gender', hue='Married', palette='husl')
    plt.title('Gender and Married')
    plt.subplot(1, 2, 2)
    sns.countplot(data=df_ori, x='Gender', hue='Loan_Status', palette='husl')
    plt.title('Gender and Loan Status')
    st.pyplot(fig2)

    st.write("## Distribution of Applicant Income ")
    income_qtiles = df_ori['ApplicantIncome'].quantile([0.25, 0.5, 0.75])
    # Definisikan batas kuartil untuk kategori pendapatan
    low_income = income_qtiles[0.25]
    medium_income = income_qtiles[0.5]
    high_income = income_qtiles[0.75]

    # Membuat fungsi untuk mengkategorikan pendapatan
    def categorize_income(income):
        if income <= low_income:
            return 'Low'
        elif income <= medium_income:
            return 'Medium'
        else:
            return 'High'

    df_ori['Income_Category'] = df_ori['ApplicantIncome'].apply(categorize_income)
    fig3 = plt.figure(figsize=(10, 8))
    sns.countplot(x='Income_Category', data=df_ori, order=['Low', 'Medium', 'High'], hue='Loan_Status', palette='husl')
    plt.xlabel('Income Category')
    plt.ylabel('Count')
    st.pyplot(fig3)

    st.write("## Distribution of Loan Amount, Loan Amount and Loan Status")
    fig4 = plt.figure(figsize=(13, 8))
    plt.subplot(1,2,1)
    sns.countplot(data=df_ori, x='Loan_Amount_Term', hue='Loan_Status', palette=('husl'))
    plt.title('Loan Amount Term and Loan Status')
    plt.xlabel('Loan Amount Term (months)')

    plt.subplot(1,2,2)
    sns.barplot(data=df, x='Loan_Amount_Term', y='LoanAmount', palette='husl', alpha=0.6)
    plt.title('Loan Amount vs Loan Amount Term')
    plt.xlabel('Loan Amount Term (months)')
    plt.ylabel('Loan Amount ')
    st.pyplot(fig4)

    st.write("## Distribution of Credit History")
    Credit_Hist = df_ori['Credit_History'].replace({0.0: 'No', 1.0: 'Yes'})
    fig5 = plt.figure(figsize=(10, 8))
    sns.countplot(data=df_ori, x=Credit_Hist, hue='Loan_Status', palette='husl')
    plt.title('Credit History and Loan Status')
    plt.xlabel('Credit History')
    plt.ylabel('Count')
    st.pyplot(fig5)

    st.write("## Distribution of Property Area dan Education")
    fig6 = plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    sns.countplot(data=df_ori, x='Property_Area', hue='Education', palette='husl')
    plt.title('Property_Area and Education')
    plt.subplot(1, 2, 2)
    sns.countplot(data=df_ori, x='Education', hue='Loan_Status', palette='husl')
    plt.title('Education and Loan Status')
    st.pyplot(fig6)
    

if __name__ == '__main__':
    run()