import streamlit as st
import pandas as pd
from PIL import Image

def run():
    
    # Judul
    st.title("Bank Loan Status Prediction")
    
    # Subheader
    st.subheader("Home")

    # Problem statement
    image = Image.open('loan.jpeg')
    st.image(image)
    st.write("Loan Status digunakan untuk melihat apakah pinjaman applicant diterima atau tidaknya oleh Bank ")
    st.write("Prediksi akan dilakukan dengan model terbaik, yang dipilih dari 5 algoritma: K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Random Forest Classifier, Decision Tree Classifier, Gradient Boosting. Penentuan model terbaik menggunakan metode Cross Validation, yang dilanjutkan dengan Hyperparameter Tuning untuk mengoptimalkan model.")
    st.markdown("---")

    # Dataset
    st.write("**Dataset**")
    st.write("Dataset diambil dari Kaggle Dataset, yang dapat diakses pada [link](https://www.kaggle.com/datasets/bhavikjikadara/loan-status-prediction/data).")
    st.write("Ada 381 Data dengan 13 feature seperti terlihat di bawah, dengan schema data di bawahnya:")
    df = pd.read_csv('loan_data.csv')
    st.dataframe(df)
    image1 = Image.open('tabel.JPG')
    st.image(image1)

if __name__ == "__main__":
  run()