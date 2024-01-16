import os
import pandas as pd
from rdkit import Chem

from torch.utils import data as torch_data

from torchdrug import data, utils, transforms
from torchdrug.core import Registry as R

@R.register("datasets.BindingDBKd")
@utils.copy_args(data.ProteinLigandDataset.load_sequence)
class BindingDBKd(data.ProteinLigandDataset):
    """ 
    Input: 
        path = "../../data/dta-datasets/tdc/"
        bindingdb_dataset.csv ==> pd.series containing the three col for usage!
        ["Drug"], ["Target"] and ["Y"] ==> .tolist() for input into the 
        load_sequence(self, sequences, smiles, targets, num_samples)

    BindingDB is a public, web-accessible database of measured binding affinities, 
    focusing chiefly on the interactions of protein considered to be drug-targets with small, drug-like molecules.

    Statistics:
        - #Molecule: 10,665
        - #Protein: 1,413
        - #Interaction: 52,284
        - split:
        train:valid:test = 4:1:1 (and test shoule be hold out)
        - #Train: 17,182
        - #Valid: 4,295
        - #Test: 4,295

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """
    # path = 'D:\Code\CurrentWork\Demo\DTA_Work\data\dta-datasets\tdc'
    splits = ["train", "valid", "holdout_test"]
    target_fields = ["affinity"]
    num_samples = [17182, 4295, 4295]

    def __init__(self, path, verbose=1, **kwargs):
        """
        - Use the tdc bindingdb dataset to create the local file.
        - Then init the bindingdb dataset instance by loading the local file to pd.DataFrame.
        - Select the col of the df to build the dataset object.
        """
        # ============================== file path existance check ============================== #        
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        # ============================== load the local csv file ============================== #
        bindingdb_path = self.path + 'bindingdb_datasets.csv'
        bindingdb_df = pd.read_csv(bindingdb_path)
        # ============================== get the col of the df ============================== #
        bindingdb_smiles = bindingdb_df["Drug"]
        bindingdb_sequences = bindingdb_df["Target"]    
        bindingdb_affinities = bindingdb_df["Y"]

        drug = bindingdb_smiles.tolist()
        protein = bindingdb_sequences.tolist()
        
        targets = {}
        targets["affinity"] = bindingdb_affinities.tolist()

        # ============================== load dataset to graph object ============================== #
        self.load_sequence(protein, drug, targets, self.num_samples, verbose=verbose, **kwargs)

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

    def get_item(self, index):
        if self.lazy:
            graph1 = data.Protein.from_sequence(self.sequences[index], **self.kwargs)
            mol = Chem.MolFromSmiles(self.smiles[index])
            if not mol:
                graph2 = None
            else:
                graph2 = data.Molecule.from_molecule(mol, **self.kwargs)
        else:
            graph1 = self.data[index][0]
            graph2 = self.data[index][1]
        item = {"graph1": graph1, "graph2": graph2}
        item.update({k: v[index] for k, v in self.targets.items()})
        if self.transform:
            item = self.transform(item)
        return item
