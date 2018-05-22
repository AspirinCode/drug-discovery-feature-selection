import pandas
import csv

output_file = '../dataset/dataset.csv'

# File dataset
drug_file = '../dataset/pubchem-compound-active-hiv1-protease.csv'
decoy_file = '../dataset/decoys.csv'

# Baca file
drug = pandas.read_csv(drug_file, dtype={'Name': str}, index_col=0)
decoy = pandas.read_csv(decoy_file, dtype={'Name': str}, index_col=0)

# Menentukan kelas
drug['Class'] = 1
decoy['Class'] = 0

# Mengambil decoy sejumlah drug secara random, sehingga dataset menjadi balance
jml_drug = len(drug)
decoy = decoy.drop_duplicates()
decoy_subset = decoy.sample(n=jml_drug)

# Menggabungkan kedua dataset dan mengisi nilai yg kosong dengan 0
dataset = pandas.concat([drug, decoy_subset]).sample(frac=1)
dataset.fillna(value=0, inplace=True)
#dataset.dropna(axis=0)

# Simpan hasilnya untuk digunakan pada langkah selanjutnya
dataset.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
