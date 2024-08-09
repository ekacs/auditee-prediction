import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

# Load the trained model
load_model = pickle.load(open('ml-prediksi-upt-v2.pkl', 'rb'))

# Load your test data
# Assuming you have a test dataset saved in 'dataset_master_awal.xlsx' (replace with your actual data)
test_data = pd.read_excel('dataset_master_awal.xlsx')

# Pilih fitur dan target
features = ['CR', 'ER', 'IGR', 'EGR']  # Nama kolom fitur
target = 'level mod1 (hanya 2 opsi)'  # Nama kolom target

# Pisahkan fitur (X) dan target (y)
X_test = test_data[features]
y_test = test_data[target]

# Konversi target ke format numerik
# Misalnya, 'berpotensi' menjadi 1 dan 'tidak berpotensi' menjadi 0
y_test = y_test.map({'berpotensi': 1, 'tidak berpotensi': 0})

# Prediksi variabel target
y_pred = load_model.predict(X_test)

# Hitung akurasi
accuracy = 100*accuracy_score(y_test, y_pred)

#print akurasi
print("akurasi model 'ml-prediksi-upt-v2.pkl' sebesar", accuracy)