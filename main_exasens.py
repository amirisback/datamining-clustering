import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

COLUMN_0 = "diagnosis"
COLUMN_1 = "id"
COLUMN_2 = "imaginary_part_min"
COLUMN_3 = "imaginary_part_avg"
COLUMN_4 = "real_part_min"
COLUMN_5 = "real_part_avg"
COLUMN_6 = "gender"
COLUMN_7 = "age"
COLUMN_8 = "smoking"

DATA_SET_FEATURES = [COLUMN_0, COLUMN_2, COLUMN_3, COLUMN_4, COLUMN_5]

# --- Membaca Data ---
raw_data_set = pd.read_csv("exasens.csv")
raw_data_set.head()

# # --- Menghilangkan Kolom Yang Tidak Perlu ---
raw_data_set = raw_data_set.drop([COLUMN_1, COLUMN_2, COLUMN_3, COLUMN_4, COLUMN_5], axis=1)
raw_data_set.head()

label_encoder = LabelEncoder()
raw_data_set[DATA_SET_FEATURES[0]] = label_encoder.fit_transform(raw_data_set[DATA_SET_FEATURES[0]])

# print(raw_data_set)

# -- Menentukan variabel yang akan di klusterkan ---
raw_data_set_x = raw_data_set.iloc[:, 0:7]
raw_data_set_x.head()

print(raw_data_set_x)

# --- Memvisualkan persebaran data ---
plt.scatter(raw_data_set.age, raw_data_set.smoking, s=10, c="c", marker="o", alpha=1)
plt.show()

# --- Mengubah Variabel Data Frame Menjadi Array ---
x_array = np.array(raw_data_set_x)
print(x_array)

# --- Menstandarkan Ukuran Variabel ---
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x_array)

# --- Menentukan dan mengkonfigurasi fungsi kmeans ---
kmeans = KMeans(n_clusters=3, random_state=123)

# --- Menentukan kluster dari data ---
kmeans.fit(x_scaled)

# --- Menampilkan pusat cluster ---
print(kmeans.cluster_centers_)

# --- Menampilkan Hasil Kluster ---
print(kmeans.labels_)

# --- Menambahkan Kolom "kluster" Dalam Data Frame Driver ---
raw_data_set["kluster"] = kmeans.labels_

# --- Memvisualkan hasil kluster ---
output = plt.scatter(x_scaled[:, 0], x_scaled[:, 1], s=100, c=raw_data_set.kluster, marker="o", alpha=1, )
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=1, marker="s");
plt.title("Hasil Klustering K-Means")
plt.colorbar(output)
plt.show()
