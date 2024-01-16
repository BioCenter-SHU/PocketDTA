import os
import json
import random
from tqdm import tqdm
import pickle
import pandas as pd
from rdkit import Chem
from copy import deepcopy

import torch
from torch.utils import data as torch_data
from torchdrug import data, utils
from torchdrug.core import Registry as R

@R.register("datasets.KIBA")
@utils.copy_args(data.ProteinLigandDataset.load_sequence)
class KIBA(data.ProteinLigandDataset):
    """ 
    Input: 
        path = "../../data/dta-datasets/KIBA/"
        method = 'sequence' or 'pdb' for different init instance methods.
    KIBA_datasets.csv ==> pd.series containing the all the col for usage!
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
        - #Molecule: 2111
        - #Protein: 229
        - #Interaction: 118254
        - split:
        train:valid:test = 4:1:1 (and test shoule be hold out)
        - #Train: 78836
        - #Valid: 19709
        - #Test: 19709
    Parameters:
        path (str): path to store the dataset 
        drug_method (str): Drug loading method, from 'smile','2d' or '3d'. 
        protein_method (str): Protein loading method, from 'sequence','pdb'.
        transform (Callable, optional): protein sequence transformation function
        lazy (bool, optional): if lazy mode is used, the protein-ligand pairs are processed in the dataloader.
            This may slow down the data loading process, but save a lot of CPU memory and dataset loading time.
        **kwargs
    """
    splits = ["train", "valid", "test"]
    target_fields = ["affinity"]
    num_samples = [78836, 19709, 19709] 
    # Three init way, first the normal way using sequences,
    # another using the pdb/sdf for protein smiles for drug
    def __init__(self, path='../../data/dta-datasets/KIBA/', drug_method='smile', protein_method='sequence',
                transform=None, lazy=False, **kwargs):
        """
        - Use the DeepDTA KIBA dataset to create the local file.
        - Then init the KIBA dataset instance by loading the local file to pd.DataFrame.
        - Select the col of the df to build the dataset object.
        """   
        # ============================== Path check and generate ============================== # 
        # os.path.expanduser expand the ~ to the full path
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.lazy = lazy
        self.transform = transform
        # get different csv file
        local_file = self.path + "kiba_datasets.csv"
        dataset_df = pd.read_csv(local_file)
        self.sequences = dataset_df["Target"].tolist()
        # ============================== Get the col info of the Dataframe ============================== #
        drug_list = dataset_df["Drug_Index"].tolist() # for further index the drug in 68 durg list
        protein_list = dataset_df["protein_Index"].tolist()  # for further index the protein in 442 protein list
        self.pdb_files = dataset_df["PDB_File"].tolist()  # pdb file name like AF-Q2M2I8-F1-model_v4.pdb
        self.smiles = dataset_df["Drug"].tolist()
        self.targets = {}  # follow the torchdrug form create a dict to store
        self.targets["affinity"] = dataset_df["Y"].tolist()  
        # ============================== Generating the self.data list [protein, mol] ============================== #
        num_sample = len(self.smiles)                
        for field, target_list in self.targets.items():
            if len(target_list) != num_sample:
                raise ValueError("Number of target `%s` doesn't match with number of molecules. "
                                 "Expect %d but found %d" % (field, num_sample, len(target_list)))

        self.data = []
        # ============================== Loading Protein ============================== #
        protein_pkl = path + protein_method + "_Protein.pkl"
        # ============================== Pkl file not exist, Creating one ==============================  #
        if not os.path.exists(protein_pkl):
            protein_file = self.path + "kiba_proteins.csv"
            if protein_method == "pdb" or "sequence" or "pocket":
                self.load_protein(protein_file, protein_method, **kwargs)
            else:
                raise ValueError("Protein method should come from 'pdb', 'sequence' or 'pocket' !")
        # ============================== Loading Pkl file ==============================  #
        with utils.smart_open(protein_pkl, "rb") as fin:
            target_protein  = pickle.load(fin)
        # ============================== Loading Drug ============================== #
        drug_pkl = path + drug_method + "_Molecule.pkl"
        if not os.path.exists(drug_pkl):
            drug_file = self.path + "kiba_ligands.csv"
            if drug_method == "smile" or "2d" or "3d":
                self.load_drug(drug_file, drug_method, **kwargs)
            else:
                raise ValueError("Drug method should be 'smile' , '2d' or '3d' !")
        with utils.smart_open(drug_pkl, "rb") as fin:
            drug_molcule  = pickle.load(fin)
        indexes = range(num_sample)
        indexes = tqdm(indexes, "Constructing Dataset from pkl file: ")
        for i in indexes:
            protein = target_protein[protein_list[i]]
            mol = drug_molcule[drug_list[i]]
            self.data.append([protein, mol])
        # ============================== Dataset Completing! ============================== #

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
    def random_split(self):  
        train_set, valid_set, test_set = torch.utils.data.random_split(self, self.num_samples)
        return train_set, valid_set, test_set 

    # fixed index as the DeepDTA provided, we need to reproduce the 5 folds for training, 1 fold for test
    def deepdta_split(self, fold):
        train_valid_index = json.load(open("../../data/dta-datasets/KIBA/train_fold_setting1.txt"))
        test_index = json.load(open("../../data/dta-datasets/KIBA/test_fold_setting1.txt"))
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

    # align the kiba index with pocket index
    def pocket_split(self, method='random', fold=None):
        '''
        KIBA Statistics:
            - #Molecule: 2,111
            - #Protein: 229
            - #Interaction: 118,254
            - split:
            train:valid:test = 4:1:1 (and test shoule be hold out)
            - #Train: 78,836
            - #Valid: 19,709
            - #Test: 19,709
        Pocket Statistics:
            - #Molecule: 2,111
            - #Pocket: 731
            - #Interaction: 356,403
            deepdta split:
            - #Train: 238,211
            - #Valid: 58,917
            - #Test: 59,275
        '''
        dataset_file = "../../data/dta-datasets/KIBA/kiba_pocket_datasets.csv"
        dataset_df = pd.read_csv(dataset_file)
        if method == 'random':
            # first random split the 118,254 interaction from kiba
            dataset_index = list(range(sum(self.num_samples)))
            # shuffle the list in place
            random.shuffle(dataset_index)
            train_index = dataset_index[:self.num_samples[0]]
            valid_index = dataset_index[self.num_samples[0]:sum(self.num_samples[:2])]
            test_index = dataset_index[sum(self.num_samples[:2]):]
            pocket_train_index = []
            for index in train_index:
                pocket_train_index.extend(dataset_df[dataset_df['interaction_index']==index].index.tolist())
            pocket_valid_index = []
            for index in valid_index:
                pocket_valid_index.extend(dataset_df[dataset_df['interaction_index']==index].index.tolist())
            pocket_test_index = []
            for index in test_index:
                pocket_test_index.extend(dataset_df[dataset_df['interaction_index']==index].index.tolist())
            train_set = torch.utils.data.Subset(self, pocket_train_index)
            valid_set = torch.utils.data.Subset(self, pocket_valid_index)
            test_set = torch.utils.data.Subset(self, pocket_test_index)
        elif method == 'deepdta':
            train_index_file = "../../data/dta-datasets/KIBA/train_pocket_index.txt"
            test_index_file = "../../data/dta-datasets/KIBA/test_pocket_index.txt"
            if not os.path.exists(train_index_file):
                # if the new text file not exist, we need to build from kiba fold text
                train_valid_index = json.load(open("../../data/dta-datasets/KIBA/train_fold_setting1.txt"))
                test_index = json.load(open("../../data/dta-datasets/KIBA/test_fold_setting1.txt"))
                # get the each fold and transform the index,like [[fold1], [fold2], [fold3], [fold4], [fold5]]
                pocket_train_valid_index = []
                for fold in range(5):
                    pocket_fold_index = []
                    for index in train_valid_index[fold]:
                        pocket_fold_index.extend(dataset_df[dataset_df['interaction_index']==index].index.tolist())
                    pocket_train_valid_index.append(pocket_fold_index)
                with open(train_index_file,'w',encoding='utf-8') as f:
                    json.dump(pocket_train_valid_index, f)
                pocket_test_index = []
                for index in test_index:
                    pocket_test_index.extend(dataset_df[dataset_df['interaction_index']==index].index.tolist())
                with open(test_index_file,'w',encoding='utf-8') as f:
                    json.dump(pocket_test_index, f)
            # use the existing new pocket text to generate Subset
            train_valid_index = json.load(open(train_index_file))
            test_index = json.load(open(test_index_file))
            print(f'==================== Training on Fold: {fold} ====================')
            train_index = []
            valid_index = []
            valid_index = train_valid_index[fold]  # which is the valid list for dataset build
            otherfolds = deepcopy(train_valid_index)  # copy a new list to not effect the raw list
            otherfolds.pop(fold)  # pop out the valid fold and left the other 4 folds for training
            for train_fold in otherfolds:  # to merge the 4 fold into a single train_index list
                train_index.extend(train_fold)
            # Get the list, now build the dataset
            test_set = torch.utils.data.Subset(self, test_index)
            valid_set = torch.utils.data.Subset(self, valid_index) 
            train_set = torch.utils.data.Subset(self, train_index)
        else:
            raise ValueError("Split method should come from 'random' or 'deepdta' !")
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
        # Protein transform operation
        if self.transform:
            item = self.transform(item)
        return item


