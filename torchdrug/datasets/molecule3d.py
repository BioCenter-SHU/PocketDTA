import os
import pickle
import csv
import math
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import logging
from rdkit.Chem import AllChem

import torch
from torchdrug import data, utils
from torchdrug.data import feature
from torchdrug.core import Registry as R
logger = logging.getLogger(__name__)

@R.register("datasets.Molecule3D")
class Molecule3D(data.MoleculeDataset):
    """
    Build the dataset Molecule with node position from the smiles file.

    Statistics:
        QM9
        - #Molecule: 133,885
        - #Regression task: 12
        - #target_fields: ["mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", "u298", "h298", "g298"]
        ZINC250K
        - #Molecule: 249,455
        - #Regression task: 2 
        - #target_fields: ["logP", "qed"]
        ZINC2m
        - #Molecule: 2,000,000
    Parameters:
        path (str): path to store the dataset
        datasetname (str): the dataset name from QM9, ZINC250K
        **kwargs
    """
    
    def __init__(self, path="../../data/molecule-datasets/", datasetname="QM9", transform=None, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.transform = transform
        self.targets = defaultdict(list)
        self.smiles_list = []
        self.data = []
        if datasetname == 'QM9':
            csv_file = self.path + "qm9.csv"
            pkl_file = self.path + "QM9.pkl.gz"
            smiles_field = "smiles"
            target_fields = ["mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", "u298", "h298", "g298"]
            smiles_list, targets = self.load_csv(csv_file=csv_file, smiles_field=smiles_field, target_fields=target_fields)
        elif datasetname == 'ZINC250K':
            csv_file = self.path + "250k_rndm_zinc_drugs_clean_3.csv"
            pkl_file = self.path + "ZINC250K.pkl.gz"
            smiles_field = "smiles"
            target_fields = ["logP", "qed"]
            smiles_list, targets = self.load_csv(csv_file=csv_file, smiles_field=smiles_field, target_fields=target_fields)
        elif datasetname == "ZINC2m":
            # the csv file is to big with only smiles info, we can not edit from app. So use different method.
            csv_file = self.path + "zinc2m_smiles.csv"
            pkl_file = self.path + "ZINC2m.pkl.gz"
            with open(csv_file, "r") as fin:
                reader = csv.reader(fin)
                reader = iter(tqdm(reader, "Loading %s" % path, utils.get_line_count(csv_file)))
                smiles_list = []
                for idx, values in enumerate(reader):
                    smiles = values[0]
                    smiles_list.append(smiles)
            targets = {}
        
        else:
            logger.warning("Chooese the dataset from 'QM9','ZINC250K' or 'ZINC2m'")

        self.targets = targets
        self.load_dataset(pkl_file, smiles_list)
    
    def load_dataset(self, pkl_file, smiles_list):
        if not os.path.exists(pkl_file):
            molecule_list = []
            # From the smiles to mol list
            smiles = tqdm(smiles_list, "Constructing Molecule from Smiles with 3D atom position: ")
            for smile in smiles:
                mol = Chem.MolFromSmiles(smile)
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
                # mol = Chem.RemoveHs(mol)
                if not mol:
                    logger.debug("Can't construct molecule from SMILES `%s`. Ignore this sample." % smile)
                    continue
                d = data.Molecule.from_molecule(mol, atom_feature="default", bond_feature="default")
                with d.node():
                    d.node_position = torch.tensor([feature.atom_position(atom) for atom in mol.GetAtoms()])
                molecule_list.append(d)
            print("Start Dumping the pkl file...")
            with utils.smart_open(pkl_file, "wb") as fout:
                num_sample = len(molecule_list)
                pickle.dump((num_sample), fout)

                indexes = range(num_sample)
                indexes = tqdm(indexes, "Dumping to %s" % pkl_file)
                for i in indexes:
                    pickle.dump((smiles_list[i], molecule_list[i]), fout)
            print("PKL Completin...")
        else:
            # ============================== PKL Loading! ============================== #
            print("Start Loading the pkl file...")
            with utils.smart_open(pkl_file, "rb") as fin:
                num_sample = pickle.load(fin)

                indexes = range(num_sample)
                indexes = tqdm(indexes, "Loading %s" % pkl_file)
                for i in indexes:
                    smile, mol= pickle.load(fin)
                    self.smiles_list.append(smile)
                    self.data.append(mol)
            print("Dataset Completing...")

    def load_csv(self, csv_file, smiles_field="smiles", target_fields=None):
        if target_fields is not None:
                target_fields = set(target_fields)
        with open(csv_file, "r") as fin:
            reader = csv.reader(fin)
            reader = iter(tqdm(reader, "Loading %s" % csv_file, utils.get_line_count(csv_file)))
            fields = next(reader)
            smiles = []
            targets = defaultdict(list)
            for values in reader:
                if not any(values):
                    continue
                if smiles_field is None:
                    smiles.append("")
                for field, value in zip(fields, values):
                    if field == smiles_field:
                        smiles.append(value)
                    elif target_fields is None or field in target_fields:
                        value = utils.literal_eval(value)
                        if value == "":
                            value = math.nan
                        targets[field].append(value)
        return smiles, targets


