# Hyperparameter-tuning-in-XGBoost-using-genetic-algorithm
We will use genetic algorithm for hyperparameter tuning in XGBoost. The dataset is from https://archive.ics.uci.edu/ml/machine-learning-databases/musk/.
It contains a set of 102 molecules, out of which 39 are identified by humans as 
having odor that can be used in perfumery and 69 having not the desired odor.
The dataset contains 6,590 low-energy conformations of these molecules, contianing 166 features.

## Install ##

For GPU enabled computers:
```
conda xgboostga create -f environment.yml
```

or create from scratch:
```
conda create --name xgboostga python=3
conda activate xgboostga
conda install jupyter matplotlib 
conda install -c conda-forge mlflow
```
XGBoost GPU:
```
conda install -c anaconda py-xgboost-gpu 
```
For CPU only:
```
conda install -c conda-forge xgboost 
```
