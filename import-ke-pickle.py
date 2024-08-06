import xgboost as xgb
import pandas as pd
import pickle

# Muat data dari file Excel
file_path = 'dataset_master_awal.xlsx'  # Path ke file Excel
data_train = pd.read_excel(file_path)

# Pilih fitur dan target
features = ['CR', 'ER', 'IGR', 'EGR']  # Nama kolom fitur
target = 'level mod1 (hanya 2 opsi)'  # Nama kolom target

X_train = data_train[features]
y_train = data_train[target]

# Konversi target ke format numerik
# Misalnya, 'berpotensi' menjadi 1 dan 'tidak berpotensi' menjadi 0
y_train = y_train.map({'berpotensi': 1, 'tidak berpotensi': 0})

# Latih model
model = xgb.XGBClassifier(eval_metric='mlogloss')
model.fit(X_train, y_train)

# Simpan model ke file pickle
with open('ml-prediksi-upt-v2.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model telah dilatih dan disimpan ke 'ml-prediksi-upt-v2.pkl'")