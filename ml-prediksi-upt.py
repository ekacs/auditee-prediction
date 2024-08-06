import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression #model logistic regression
from sklearn.ensemble import RandomForestClassifier #model random forest
from sklearn.neighbors import KNeighborsClassifier #model KNN

st.write("""
# AUDITEE PREDICTION APP 

The purpose of this application is to predict which auditees are most likely to uncover state financial losses. The output of this application can then be used to inform the selection of audit targets. 
[(Would you like to learn more about our article regarding this?)](https://bit.ly/Research_ekacs).

Please be informed that the results of [Mr. Eka CS's thesis research](https://bit.ly/thesis_MrEka) were used as the primary reference for developing this machine learning model.

""")

# Displays the user input features
st.subheader('User Input features')

st.write("""Awaiting Excell file to be uploaded. Currently using example input parameters. [click for example XLSX input file](https://docs.google.com/spreadsheets/d/1-aQCD8iFIUuxkqZo1rvTl2mjoVVJAjL5?rtpof=true&usp=drive_fs)""")


# Reads in saved classification model
from sklearn.ensemble import VotingClassifier
model1 = LogisticRegression()
model2 = RandomForestClassifier(n_estimators=100)
model3 = KNeighborsClassifier(n_neighbors=3)
# Gabungkan model-model dalam ensemble dengan teknik voting
ensemble_model = VotingClassifier(estimators=[('model1', model1),
                                              ('model2', model2),
                                              ('model3', model3)],
                                  voting='soft')
load_model = pickle.load(open('ml-prediksi-upt.pkl', 'rb'))

# Collects user input features into dataframe
uploaded_file = st.file_uploader("Upload your input excell file", type=["xlsx"])

if uploaded_file is not None:
    input_df = pd.read_excel(uploaded_file)
    #st.write(input_df)
    # Definisikan fungsi untuk menerapkan rumus Excel
    # cash ratio
    def CR(row):
        if row['Kewajiban jangka pendek'] != 0:
            return (row['kas'] + row['setara kas']) / row['Kewajiban jangka pendek']
        else:
            return 0


    # Terapkan rumus pada DataFrame menggunakan fungsi 'apply'
    input_df['CR'] = input_df.apply(CR, axis=1)


    # efisiensi ratio
    def ER(row):
        if row['Realisasi Penerimaan'] != 0:
            return row['Realisasi Pengeluaran'] / row['Realisasi Penerimaan']
        else:
            return 0


    # Terapkan rumus pada DataFrame menggunakan fungsi 'apply'
    input_df['ER'] = input_df.apply(ER, axis=1)


    # income growth ratio
    def IGR(row):
        if row['Penerimaan Tahun sebelumnya'] != 0:
            return (row['Realisasi Penerimaan'] - row['Penerimaan Tahun sebelumnya']) / row[
                'Penerimaan Tahun sebelumnya']
        else:
            return 0


    # Terapkan rumus pada DataFrame menggunakan fungsi 'apply'
    input_df['IGR'] = input_df.apply(ER, axis=1)


    # expense growth ratio
    def EGR(row):
        if row['Pengeluaran Tahun sebelumnya'] != 0:
            return (row['Realisasi Pengeluaran'] - row['Pengeluaran Tahun sebelumnya']) / row[
                'Pengeluaran Tahun sebelumnya']
        else:
            return 0


    # Terapkan rumus pada DataFrame menggunakan fungsi 'apply'
    input_df['EGR'] = input_df.apply(ER, axis=1)


    # Definisikan fungsi untuk menerapkan rumus Excel
    # cash ratio
    def CR(row):
        if row['Kewajiban jangka pendek'] != 0:
            return (row['kas'] + row['setara kas']) / row['Kewajiban jangka pendek']
        else:
            return 0


    # Terapkan rumus pada DataFrame menggunakan fungsi 'apply'
    input_df['CR'] = input_df.apply(CR, axis=1)


    # efisiensi ratio
    def ER(row):
        if row['Realisasi Penerimaan'] != 0:
            return row['Realisasi Pengeluaran'] / row['Realisasi Penerimaan']
        else:
            return 0


    # Terapkan rumus pada DataFrame menggunakan fungsi 'apply'
    input_df['ER'] = input_df.apply(ER, axis=1)


    # income growth ratio
    def IGR(row):
        if row['Penerimaan Tahun sebelumnya'] != 0:
            return (row['Realisasi Penerimaan'] - row['Penerimaan Tahun sebelumnya']) / row[
                'Penerimaan Tahun sebelumnya']
        else:
            return 0


    # Terapkan rumus pada DataFrame menggunakan fungsi 'apply'
    input_df['IGR'] = input_df.apply(ER, axis=1)


    # expense growth ratio
    def EGR(row):
        if row['Pengeluaran Tahun sebelumnya'] != 0:
            return (row['Realisasi Pengeluaran'] - row['Pengeluaran Tahun sebelumnya']) / row[
                'Pengeluaran Tahun sebelumnya']
        else:
            return 0

    # Terapkan rumus pada DataFrame menggunakan fungsi 'apply'
    input_df['EGR'] = input_df.apply(ER, axis=1)

    #ambil data uji variabel x saja dari data input

    x_test = input_df.drop(['Kode Auditee', 'Nama Auditee', 'Tahun', 'Provinsi', 'Kewajiban jangka pendek', 'kas',
                            'setara kas', 'Realisasi Pengeluaran', 'Realisasi Penerimaan',
                            'Pengeluaran Tahun sebelumnya', 'Penerimaan Tahun sebelumnya'], axis=1)


    # Predict the target variable
    prediksi = load_model.predict(x_test)
    # Add the predictions to the DataFrame
    input_df['prediksi'] = prediksi
    # show dataframe
    st.write(input_df)


    def draw_pie_chart(data_frame):
        # Menghitung frekuensi nilai dalam kolom
        value_counts = data_frame['prediksi'].value_counts()

        # Buat pie chart
        fig, ax = plt.subplots()
        ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Membuat aspek ratio menjadi sama untuk mendapatkan lingkaran

        # Tampilkan pie chart
        st.pyplot(fig)


    def main():
        st.write(f"Jumlah UPT: {len(input_df)}")
        st.write(f"Jumlah UPT yang berpotensi: {len(input_df.loc[input_df['prediksi']=='berpotensi'])}")
        st.write(f"Jumlah UPT yang tidak berpotensi: {len(input_df.loc[input_df['prediksi'] == 'tidak berpotensi'])}")
              # Panggil fungsi untuk menggambar pie chart berdasarkan kolom
        draw_pie_chart(input_df)
    if __name__ == "__main__":
        main()