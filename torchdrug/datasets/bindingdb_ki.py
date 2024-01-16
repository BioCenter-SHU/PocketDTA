import os
import json
import pandas as pd
from rdkit import Chem
from copy import deepcopy
import torch
from torch.utils import data as torch_data
from torchdrug import data, utils, transforms
from torchdrug.core import Registry as R

@R.register("datasets.BindingDBKi")
@utils.copy_args(data.ProteinLigandDataset.load_sequence)
class BindingDBKi(data.ProteinLigandDataset):
    """ 
    Input: 
        path = "../../data/dta-datasets/BindingDB/"
        method = 'sequence' or 'pdb' for different init instance methods.
    davis_datasets.csv ==> pd.series containing the all the col for usage!
    ["Drug"], ["Target"], ["Y"] and ["PDB_File"] ==> .tolist() for input into the 
    load_sequence/load_pdb_smile function to build the dataset containing attritube
    Notes: the class attributes do not need to be defined first. Once used they will be defined.
    - pdb_files: for protein 3d pdb file name and location
    - sequences: for protein 1d sequence
    - smiles: for drug 1d smiles
    - targets: for the affinity scores
    - data: for the torchdrug tuple(Protein, Molecule)
    The interaction of 68(filtered protein under 10 interactions) kinase inhibitors with 442 kinases 
    covering>80% of the human catalytic protein kinome.(only 379 kinases are unique).
    Statistics:
        - #Molecule: 159,633
        - #Protein: 2,397
        - #Interaction: 296,151
        - split:
        train:valid:test = 4:1:1 (and test shoule be hold out)
        - #Train: 197435
        - #Valid: 49358
        - #Test: 49358
    Parameters:
        path (str): path to store the dataset 
        method (str): method to build the datasets
        verbose (int, optional): output verbose level
        **kwargs
    """
    splits = ["train", "valid", "holdout_test"]
    target_fields = ["affinity"]
    # Two init way, first the normal way using sequences,
    # another using the pdb for protein smiles for drug
    def __init__(self, path, method, verbose=1, **kwargs):
        """
        - Use the TDC dataset to create the local file.
        - Then init the BindingDB dataset instance by loading the local file to pd.DataFrame.
        - Select the col of the df to build the dataset object.
        """     
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        # ============================== load the local file is faster ============================== #
        local_file = self.path + 'bindingdb_ki_datasets.csv'
        self.num_samples = [197435, 49358, 49358] 
        self.dataset_df = pd.read_csv(local_file)
        # ============================== get the col of the df ============================== #
        davis_smiles = self.dataset_df["Drug"]
        davis_sequences = self.dataset_df["Target"]    
        davis_affinities = self.dataset_df["Y"]
        davis_pdb = self.dataset_df["PDB_File"]
        
        self.smiles = davis_smiles.tolist()
        self.sequences = davis_sequences.tolist()
        self.targets = {} 
        self.targets["affinity"] = davis_affinities.tolist()
        self.pdb_files = davis_pdb.tolist()
        if method == 'pdb':
            self.load_pdb_smile(self.path, self.pdb_files, self.smiles, self.targets, self.num_samples, verbose=verbose, **kwargs)
            # pkl_file = os.path.join(self.path, 'davis_dataset.pkl.gz')
            # ============================== path and pkl file check for quick loading ============================== #   
            # if os.path.exists(pkl_file):
                # self.load_pickle(pkl_file, verbose=verbose, **kwargs)
            # else:
                # ============================== pdb method is slow so pkl is needed ============================== #
                # self.load_pdb_smile(self.path, self.pdb_files, self.smiles, self.targets, self.num_samples, verbose=verbose, **kwargs)
                # self.save_pickle(pkl_file, verbose=verbose)  # the dumping processing is to slow
        elif method == 'sequence':
            # ============================== load dataset to graph object ============================== #
            self.load_sequence(self.sequences, self.smiles, self.targets, self.num_samples, verbose=verbose, **kwargs)
        else:
            print("method should be 'sequence' or 'pdb' for dataset init!")

    # sequence split
    def split(self, keys=None):  
        keys = keys or self.splits
        offset = 0
        splits = []
        for split_name, num_sample in zip(self.splits, self.num_samples):
            if split_name in keys:
                split = torch_data.Subset(self, range(offset, offset + num_sample))
                splits.append(split)
            offset += num_sample
        return splits

    # random split 4:1:1 as the num_samples
    def random_split(self, keys=None):  
        train_set, valid_set, test_set = torch.utils.data.random_split(self, self.num_samples)
        return train_set, valid_set, test_set 

    # fixed index as the DeepDTA provided, we need to reproduce the 5 folds for training, 1 fold for test
    def deepdta_split(self, fold):
        train_valid_index = json.load(open("../../data/dta-datasets/Davis/train_fold_setting1.txt"))
        test_index = json.load(open("../../data/dta-datasets/Davis/test_fold_setting1.txt"))
        test_set = torch.utils.data.Subset(self, test_index)
        print(f'==================== Training on Fold: {fold} ====================')
        train_index = []
        valid_index = []
        valid_index = train_valid_index[fold]  # which is the valid list for dataset build
        otherfolds = deepcopy(train_valid_index)  # copy a new list to not effect the raw list
        otherfolds.pop(fold)  # pop out the valid fold and left the other 4 folds for training
        for train_fold in otherfolds:  # to merge the 4 fold into a single train_index list
            train_index.extend(train_fold)
        # Get the list, now build the dataset
        valid_set = torch.utils.data.Subset(self, valid_index) 
        train_set = torch.utils.data.Subset(self, train_index)
        return train_set, valid_set, test_set 

    def get_item(self, index):
        if self.lazy:
            # graph1 = data.Protein.from_sequence(self.sequences[index], **self.kwargs)
            graph1 = data.Protein.from_pdb(self.pdb_files[index], self.kwargs)
            mol = Chem.MolFromSmiles(self.smiles[index])
            if not mol:
                graph2 = None
            else:
                graph2 = data.Molecule.from_molecule(mol, **self.kwargs)
        else:
            graph1 = self.data[index][0]
            graph2 = self.data[index][1]
        if hasattr(graph1, "residue_feature"):
            with graph1.residue():
                graph1.residue_feature = graph1.residue_feature.to_dense()
        item = {"graph1": graph1, 
                "graph2": graph2,}
        item.update({k: v[index] for k, v in self.targets.items()})
        if self.transform:
            item = self.transform(item)
        return item