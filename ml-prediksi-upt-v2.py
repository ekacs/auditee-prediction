import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from xgboost import XGBClassifier  # Mengimpor XGBClassifier dari pustaka xgboost

st.write("""
# AUDITEE PREDICTION APP 

The purpose of this application is to predict which auditees are most likely to uncover state financial losses. The output of this application can then be used to inform the selection of audit targets. 
[(Would you like to learn more about our article regarding this?)](https://bit.ly/Research_ekacs).

Please be informed that the results of [Mr. Eka CS's thesis research](https://bit.ly/thesis_MrEka) were used as the primary reference for developing this machine learning model.
""")

st.subheader('User Input features')
st.write("""Awaiting Excel file to be uploaded. Currently using example input parameters. [click for example XLSX input file](https://docs.google.com/spreadsheets/d/1-aQCD8iFIUuxkqZo1rvTl2mjoVVJAjL5?rtpof=true&usp=drive_fs)""")

# Define the XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Load the trained model from a pickle file
load_model = pickle.load(open('ml-prediksi-upt-v2.1.pkl', 'rb'))

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

    st.write(input_df)

    def draw_pie_chart(data_frame):
        value_counts = data_frame['prediksi'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

    def main():
        st.write(f"Jumlah UPT : {len(input_df)}")
        st.write(f"Jumlah UPT yang berpotensi: {len(input_df.loc[input_df['prediksi']==0])}")
        st.write(f"Jumlah UPT yang tidak berpotensi: {len(input_df.loc[input_df['prediksi'] == 1])}")
        draw_pie_chart(input_df)

    if __name__ == "__main__":
        main()