# Machine learning for compound potency prediction

--------------------- MACHINE and DEEP LEARNING SCRIPTS: GENERAL INFORMATION  ---------------------

This folder contains scripts to build and analyse ML models. 

Should be downloaded and can all Jupyter notebooks can be used inside the folder.


The content of the folders is summarized in the following.

Jupyter Notebooks (.ipynb):

(1) ml_models: Jupyter notebook with machine and deep learning models (MR, kNN, SVR, RFR, DNN), 
able to predict compound potency for different datasets (Complete/Random/Diverse sets)

(2) gcn_models: Jupyter notebook with graph neural networks models (GCN), able to predict 
compound potency for different datasets (Complete/Random/Diverse sets)

(3) ml_models_cluster_potent_sets: Jupyter notebook with machine and deep learning models 
(MR, kNN, SVR, RFR, DNN), able to predict compound potency for different datasets (Cluster/Potent sets)

(4) gcn_models_cluster_potent_sets: Jupyter notebook with graph neural networks models (GCN), 
able to predict compound potency for different datasets (Cluster/Potent sets)

(5) data_analysis_figures: Jupyter notebook with a workflow for the data analysis of compound potency 
predictions from regression models and respective figures generation.


Python scripts (.py):

(6) ml_utils: script that provide supporting functions for ML/DL models generation

(7) fingerprint: script to calculate molecular fingerprints (Morgan fingerprints)

(8) machine_learning_models: script to build ML/DL models for regression (MR, kNN, SVR, RFR, DNN)


(9) Folders:
	
	- dataset: stores the compound potency dataset used in this analysis

	- ccr_results : stores the CCR algorithm results

	- regression_results: stores regression models predictions


(10) Python environment:

	- ML_env.yml provides the python environment used for this analysis. (Requires instalation see below)


Order of Jupyter notebook execution:

1. (1), (2), (3), (4) should be run first, to generate the model predictions (results).

2. (5) should be run, afterwards, to generate the respective figures. 



Python environment installation:

1. Open Anaconda command line

2. Type 'conda env create -n ENVNAME --file ENV.yml', where 'ENVNAME' is the desired environment and 'ENV' the full path to the yml file.


Python environment export:

1. Open Anaconda command line

2. Type 'conda env export ENVNAME>ENV.yml', where 'ENVNAME' is the desired environment and 'ENV' the full path to the yml file.


Requirements:

- python=3.9
- scipy=1.8.1
- numpy=1.22.4
- scikit-learn==1.1.1
- tensorflow==2.9.1
- keras==2.9.0
- rdkit==2022.3.3
- cudatoolkit=11.2.2
- dgl-cuda11.1=0.8.1
- deepchem==2.6.1
- tqdm=4.64.0

