# About PocketDTA: A Pocket-Based Multimodal Deep Learning Model for Drug-Target Affinity Prediction

The approach used in this work is the modeling of protein sequences and compound 1D representations (SMILES) with convolutional neural networks (CNNs) to predict the binding affinity value of drug-target pairs.

Our approach used an effective form of protein input, which combines protein pocket structures with pretrained sequence embedding. In addition, a pocket-based multimodal deep learning model for drug-target affinity prediction called PocketDTA is proposed.

![Figure](https://github.com/BioCenter-SHU/PocketDTA/blob/main/figures/model.png)
# Installation

## Data

All the raw data came from the previous work. The sequence-based datasets (Filtered Davis, Davis and KIBA) and the split came from [DeepDTA](https://github.com/hkmztrk/DeepDTA) and [MDeePred](https://github.com/cansyl/MDeePred/). The structure-based dataset PDBbind and the split could be download from the Official [website](http://pdbbind.org.cn/index.php).

We customized the dataset construction file following [torchdrug](https://torchdrug.ai/) format which torchdrug did not provide (`torchdrug/datasets/davis.py`). To use our dataset file directly like, we suggest you just replace our `torchdrug`  file with you installation package(env file in you conda).  

## Requirements

You'll need to follow the TorchDrug [installation](https://torchdrug.ai/docs/installation.html) which is our main platform. The conda environment have been tested in Linux/Windows. The detailed requirement can be found in `requirements.txt`.
The main dependencies are listed below:
*  scikit-learn=1.1.3
*  torch=1.13.1+cu117
*  torchdrug=0.2.0.post1
*  torch-geometric=2.2.0
*  wandb=0.14.0

#Usage
## Dataset construction
Scripts for all four dataset construction are provided in [DatasetBuilding](https://github.com/BioCenter-SHU/PocketDTA/tree/main/script/notebook/DatasetBuilding) in the `script/notebook/DatasetBuilding/` file. Davis dataset for example, three main notebook file need for generating DTA, protein, drug pkl file.

* `DavisDataset.ipynb`
* `DavisESM.ipynb`
* `DavisMolecule.ipynb`

The file `torchdrug/datasets/davis.py` in torchdrug then provide the whole picture for Davis and Filtered Davis dataset.

## Prediction
The single sun and wandb sweep file can be found in `script/pythonfile/`, after enter your conda env just use `python SingleRun_davis.py` could run the single run on Davis dataset. The sweep need wandb package for automatic hyperparamers searching.


