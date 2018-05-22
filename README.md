# Drug Discovery Feature Selections

Virtual screening kandidat obat berbasis machine learning pada database tanaman herbal Indonesia dengan berbagai strategi feature selection (master thesis research of Rahman Pujianto, Universitas Indonesia, 2017).

## Dataset Preparation Steps

Required tools:

* OpenBabel http://openbabel.org
* PaDEL Descriptor http://www.yapcwsoft.com/dd/padeldescriptor/

Extracting positive (label = 1) training data:

1. Convert `sdf` to `mol2`: `obabel pubchem-compound-active-hiv1-protease.sdf -O dataset/pubchem-compound-active-hiv1-protease_mol2/hiv1-protease.mol2`
1. Convert `mol2` tp `csv`: `java -jar PaDEL-Descriptor.jar -2d -dir dataset/pubchem-compound-active-hiv1-protease_mol2/ -file dataset/pubchem-compound-active-hiv1-protease.csv`

Extracting negative (label = 0) training data:

1. Convert `sdf` to `mol2`: `obabel decoys_final.sdf -O dataset/decoys_final_mol2/decoys_final.mol2`
1. Convert `mol2` tp `csv`: `java -jar PaDEL-Descriptor.jar -2d -dir ataset/decoys_final_mol2/ -file dataset/decoys.csv`

Extracting test data (unlabeled):

1. Convert `mol2` tp `csv`: `java -jar PaDEL-Descriptor.jar -2d -dir dataset/HerbalDB_mol2/ -file dataset/HerbalDB.csv`


# Feature Selection

Dependency:

* Python 3.x
* Python3-tk (on ubuntu `sudo apt install python3-tk`)
* Virtualenv (optional. for isolated environment)

Dependency library installation: `pip install -r requirements.txt`

Steps:

1. Preprocessing `python 01-prepare-data.py`
1. Feature selection `python 02-feature-selection.py`