import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from xgboost import XGBClassifier  # Mengimpor XGBClassifier dari pustaka xgboost

st.write("""
# AUDITEE PREDICTION APP 

Ini adalah aplikasi untuk memprediksi Kantor Satuan Kerja yang berpotensi terdapat temuan diantaranya:
1) PNPB kurang pungut
2) Denda keterlambatan pekerjaan belum dipungut, dan/atau
3) Kelebihan pembayaran pekerjaan

Dasar prediksi ini bekerja menggunakan algoritma machine learning XGbooster dan nilai variabel indikator kinerja keuangan Kantor Satuan Kerja sebagai inputnya

Prediksi ini terinspirasi dari hasil [thesis Sdr. Eka C. Setyawan](https://bit.ly/thesis_MrEka) dan artikel rancangan pembuatan machine learning ini dapat dilihat di [karya tulis ini](https://tinyurl.com/ML-auditee-prediction)
""")

st.subheader('Masukkan nilai indikator kinerja keuangan')
st.write("""Silahkan memasukkan nilai indikator kinerja keuangan Kantor yang ingin diprediksi. [klik untuk format file inputnya](https://docs.google.com/spreadsheets/d/1-aQCD8iFIUuxkqZo1rvTl2mjoVVJAjL5?rtpof=true&usp=drive_fs)""")

# Define the XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Load the trained model from a pickle file
load_model = pickle.load(open('ml-prediksi-upt-v2.pkl', 'rb'))

uploaded_file = st.file_uploader("Upload your input Excel file", type=["xlsx"])

if uploaded_file is not None:
    input_df = pd.read_excel(uploaded_file)

    # Define functions for calculating ratios
    def CR(row):
        if row['Kewajiban jangka pendek'] != 0:
            return (row['kas'] + row['setara kas']) / row['Kewajiban jangka pendek']
        return 0

    def ER(row):
        if row['Realisasi Penerimaan'] != 0:
            return row['Realisasi Pengeluaran'] / row['Realisasi Penerimaan']
        return 0

    def IGR(row):
        if row['Penerimaan Tahun sebelumnya'] != 0:
            return (row['Realisasi Penerimaan'] - row['Penerimaan Tahun sebelumnya']) / row['Penerimaan Tahun sebelumnya']
        return 0

    def EGR(row):
        if row['Pengeluaran Tahun sebelumnya'] != 0:
            return (row['Realisasi Pengeluaran'] - row['Pengeluaran Tahun sebelumnya']) / row['Pengeluaran Tahun sebelumnya']
        return 0

    # Apply functions to calculate new features
    input_df['CR'] = input_df.apply(CR, axis=1)
    input_df['ER'] = input_df.apply(ER, axis=1)
    input_df['IGR'] = input_df.apply(IGR, axis=1)
    input_df['EGR'] = input_df.apply(EGR, axis=1)

    # Prepare features for prediction
    x_test = input_df.drop(['Kode Auditee', 'Nama Auditee', 'Tahun', 'Provinsi', 'Kewajiban jangka pendek', 'kas',
                            'setara kas', 'Realisasi Pengeluaran', 'Realisasi Penerimaan',
                            'Pengeluaran Tahun sebelumnya', 'Penerimaan Tahun sebelumnya'], axis=1)

    # Predict the target variable
    prediksi = load_model.predict(x_test)
    input_df['prediksi'] = prediksi

    # Map numerical predictions to descriptive labels
    input_df['prediksi_label'] = input_df['prediksi'].map({1: 'berpotensi', 0: 'tidak berpotensi'})

    st.write(input_df)

    def draw_pie_chart(data_frame):
        value_counts = data_frame['prediksi_label'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

    def main():
        st.write(f"Jumlah UPT : {len(input_df)}")
        st.write(f"Jumlah UPT yang berpotensi: {len(input_df.loc[input_df['prediksi_label']=='berpotensi'])}")
        st.write(f"Jumlah UPT yang tidak berpotensi: {len(input_df.loc[input_df['prediksi_label'] == 'tidak berpotensi'])}")
        draw_pie_chart(input_df)

    if __name__ == "__main__":
        main()