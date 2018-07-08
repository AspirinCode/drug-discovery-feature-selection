# Drug Discovery Feature Selections

Experiment replication of Rahman Pujianto master thesis research (Universitas Indonesia, 2017).

## Dataset Preparation

Prepared (trainable) datasets are provided in `dataset/dataset.tar.gz`. Information below are provided as an additional information on how to prepare the dataset from raw sources (`*.sdf` or `.mol2` files).

Required tools:

* OpenBabel http://openbabel.org
* PaDEL Descriptor http://www.yapcwsoft.com/dd/padeldescriptor/

Extracting positive (label = 1) training data:

1. Convert `sdf` to `mol2`: `obabel ../dataset/pubchem-compound-active-hiv1-protease.sdf -O ../dataset/pubchem-compound-active-hiv1-protease_mol2/hiv1-protease.mol2`
1. Convert `mol2` tp `csv`: `java -jar PaDEL-Descriptor.jar -2d -addhydrogens -removesalt -dir ../dataset/pubchem-compound-active-hiv1-protease_mol2/ -file ../dataset/pubchem-compound-active-hiv1-protease.csv`

Extracting negative (label = 0) training data:

1. Convert `sdf` to `mol2`: `../dataset/obabel decoys_final.sdf -O ../dataset/decoys_final_mol2/decoys_final.mol2`
1. Convert `mol2` tp `csv`: `java -jar PaDEL-Descriptor.jar -2d -addhydrogens -removesalt -dir ../dataset/decoys_final_mol2/ -file ../dataset/decoys.csv`

Extracting test data (unlabeled):

1. Convert `mol2` tp `csv`: `java -jar PaDEL-Descriptor.jar -2d -addhydrogens -removesalt -dir ../dataset/HerbalDB_mol2/ -file ../dataset/HerbalDB.csv`


## Feature Selection

Dependency:

* Python 3.x
* Python3-tk (on ubuntu `sudo apt install python3-tk`)
* Virtualenv (optional. for isolated environment)

Dependency library installation: `pip install -r requirements.txt`

Steps:

1. Extract preprocessed data from `dataset/dataset.tar.gz` (if you have raw csv data, use `python 01-prepare-data.py`)
1. Feature selection with SVM-RFE `python 02-feature-selection-svm-rfe.py` 
1. Feature selection with Wrapper Method (GA + SVM) `python 02-feature-selection-wm.py`
1. Evaluate selected features using PubChem dataset `python 03-evaluate-1.py`
1. Evaluate selected features using Indonesian Herbal dataset `python 03-evaluate-2.py`

> Evaluation scripts display accuracy scores in console, save raw results in `csv` files and display result chart(s) to screen